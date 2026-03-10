#include "openmc/distribution.h"

#include <algorithm> // for copy
#include <array>
#include <cmath>     // for sqrt, floor, max
#include <iterator>  // for back_inserter
#include <numeric>   // for accumulate
#include <stdexcept> // for runtime_error
#include <string>    // for string, stod

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// QuantileTable implementation
//==============================================================================

double QuantileTable::operator()(double u) const
{
  // Clamp to [0, 1]
  u = std::max(0.0, std::min(1.0, u));

  // Binary search for the interval containing u
  auto it = std::lower_bound(c.begin(), c.end(), u);
  int i = static_cast<int>(it - c.begin()) - 1;
  i = std::max(0, std::min(i, static_cast<int>(c.size()) - 2));

  // Linear interpolation
  double dc = c[i + 1] - c[i];
  if (dc <= 0.0)
    return x[i];
  double frac = (u - c[i]) / dc;
  return x[i] + frac * (x[i + 1] - x[i]);
}

double find_upper_bound(const std::function<double(double)>& eval_pdf,
  double x_start, double x_initial, double fraction)
{
  // Find peak value in the initial region
  constexpr int n_search = 50;
  double dx = (x_initial - x_start) / n_search;
  double f_peak = 0.0;
  for (int i = 0; i <= n_search; ++i) {
    double xi = x_start + i * dx;
    f_peak = std::max(f_peak, eval_pdf(xi));
  }

  if (f_peak == 0.0)
    return x_initial;

  double threshold = f_peak * fraction;

  // March rightward, doubling step size, until PDF < threshold
  double x_hi = x_initial;
  double step = x_initial - x_start;
  if (step <= 0.0)
    step = 1.0;

  for (int iter = 0; iter < 100; ++iter) {
    if (eval_pdf(x_hi) < threshold)
      return x_hi;
    step *= 2.0;
    x_hi += step;
  }
  return x_hi;
}

QuantileTable build_quantile_table(
  const std::function<double(double)>& eval_pdf, double x_lo, double x_hi,
  int n_initial, double tol)
{
  // Build uniform initial grid
  vector<double> xg(n_initial);
  for (int i = 0; i < n_initial; ++i) {
    xg[i] = x_lo + (x_hi - x_lo) * i / (n_initial - 1);
  }

  // Evaluate PDF on the grid
  vector<double> fg(n_initial);
  for (int i = 0; i < n_initial; ++i) {
    fg[i] = eval_pdf(xg[i]);
  }

  // Trapezoidal integration to build CDF
  vector<double> cg(n_initial);
  cg[0] = 0.0;
  for (int i = 1; i < n_initial; ++i) {
    cg[i] = cg[i - 1] + 0.5 * (fg[i - 1] + fg[i]) * (xg[i] - xg[i - 1]);
  }

  // Normalize CDF to [0, 1]
  double total = cg[n_initial - 1];
  if (total > 0.0) {
    for (int i = 0; i < n_initial; ++i) {
      cg[i] /= total;
    }
  }
  cg[n_initial - 1] = 1.0; // ensure exact 1.0 at end

  // Adaptive refinement: check midpoints for quantile accuracy
  // and insert new points where the linearly-interpolated quantile
  // differs from the true quantile by more than tol * (x_hi - x_lo)
  double abs_tol = tol * (x_hi - x_lo);
  constexpr int max_refinements = 3;
  constexpr int max_points = 10000;

  for (int round = 0; round < max_refinements; ++round) {
    vector<double> new_xg, new_fg, new_cg;
    new_xg.push_back(xg[0]);
    new_fg.push_back(fg[0]);
    new_cg.push_back(cg[0]);

    bool refined = false;
    for (std::size_t i = 0; i < xg.size() - 1; ++i) {
      double x_mid = 0.5 * (xg[i] + xg[i + 1]);
      double f_mid = eval_pdf(x_mid);

      // True CDF at midpoint (trapezoidal from left endpoint)
      double c_mid_true =
        cg[i] + 0.5 * (fg[i] + f_mid) * (x_mid - xg[i]) / total;

      // Linearly interpolated CDF at midpoint
      double c_mid_interp = 0.5 * (cg[i] + cg[i + 1]);

      // Estimate quantile error: if the CDF error is dc, and PDF at midpoint
      // is f, then quantile error ≈ dc / (f/total). But simpler: if the x
      // values where c_mid_true and c_mid_interp map back differ by more
      // than tol, refine.
      double dc = std::fabs(c_mid_true - c_mid_interp);
      double x_error = (f_mid > 0.0) ? dc * total / f_mid : 0.0;

      if (x_error > abs_tol &&
          static_cast<int>(new_xg.size() + (xg.size() - i)) < max_points) {
        // Insert midpoint
        new_xg.push_back(x_mid);
        new_fg.push_back(f_mid);
        new_cg.push_back(0.0); // will recompute
        refined = true;
      }

      new_xg.push_back(xg[i + 1]);
      new_fg.push_back(fg[i + 1]);
      new_cg.push_back(0.0); // will recompute
    }

    if (!refined)
      break;

    // Recompute CDF for the refined grid
    xg = std::move(new_xg);
    fg = std::move(new_fg);
    cg.resize(xg.size());
    cg[0] = 0.0;
    for (std::size_t i = 1; i < xg.size(); ++i) {
      cg[i] = cg[i - 1] + 0.5 * (fg[i - 1] + fg[i]) * (xg[i] - xg[i - 1]);
    }
    total = cg.back();
    if (total > 0.0) {
      for (std::size_t i = 0; i < xg.size(); ++i) {
        cg[i] /= total;
      }
    }
    cg.back() = 1.0;
  }

  // Build the QuantileTable (CDF -> x)
  QuantileTable qt;
  qt.c = std::move(cg);
  qt.x = std::move(xg);
  return qt;
}

//==============================================================================
// Helper function for computing importance weights from biased sampling
//==============================================================================

vector<double> compute_importance_weights(
  const vector<double>& p, const vector<double>& b)
{
  std::size_t n = p.size();

  // Normalize original probabilities
  double sum_p = std::accumulate(p.begin(), p.end(), 0.0);
  vector<double> p_norm(n);
  for (std::size_t i = 0; i < n; ++i) {
    p_norm[i] = p[i] / sum_p;
  }

  // Normalize bias probabilities
  double sum_b = std::accumulate(b.begin(), b.end(), 0.0);
  vector<double> b_norm(n);
  for (std::size_t i = 0; i < n; ++i) {
    b_norm[i] = b[i] / sum_b;
  }

  // Compute importance weights
  vector<double> weights(n);
  for (std::size_t i = 0; i < n; ++i) {
    weights[i] = (b_norm[i] == 0.0) ? INFTY : p_norm[i] / b_norm[i];
  }
  return weights;
}

std::pair<double, double> Distribution::sample(uint64_t* seed) const
{
  if (bias_) {
    // Sample from the bias distribution and compute importance weight
    double val = bias_->sample_unbiased(seed);
    double wgt = this->evaluate(val) / bias_->evaluate(val);
    return {val, wgt};
  } else {
    // Unbiased sampling: return sampled value with weight 1.0
    double val = sample_unbiased(seed);
    return {val, 1.0};
  }
}

// PDF evaluation not supported for all distribution types
double Distribution::evaluate(double x) const
{
  throw std::runtime_error(
    "PDF evaluation not implemented for this distribution type.");
}

// CDF evaluation not supported for all distribution types
double Distribution::cdf(double x) const
{
  throw std::runtime_error(
    "CDF evaluation not implemented for this distribution type.");
}

// Quantile evaluation not supported for all distribution types
double Distribution::quantile(double u) const
{
  throw std::runtime_error(
    "Quantile evaluation not implemented for this distribution type.");
}

void Distribution::read_bias_from_xml(pugi::xml_node node)
{
  if (check_for_node(node, "bias")) {
    pugi::xml_node bias_node = node.child("bias");

    if (check_for_node(bias_node, "bias")) {
      openmc::fatal_error(
        "Distribution has a bias distribution with its own bias distribution. "
        "Please ensure bias distributions do not have their own bias.");
    }

    UPtrDist bias = distribution_from_xml(bias_node);
    this->set_bias(std::move(bias));
  }
}

//==============================================================================
// DiscreteIndex implementation
//==============================================================================

DiscreteIndex::DiscreteIndex(pugi::xml_node node)
{
  auto params = get_node_array<double>(node, "parameters");
  std::size_t n = params.size() / 2;

  assign({params.data() + n, n});
}

DiscreteIndex::DiscreteIndex(span<const double> p)
{
  assign(p);
}

void DiscreteIndex::assign(span<const double> p)
{
  prob_.assign(p.begin(), p.end());
  this->init_alias();
}

void DiscreteIndex::init_alias()
{
  normalize();

  // The initialization and sampling method is based on Vose
  // (DOI: 10.1109/32.92917)
  // Vectors for large and small probabilities based on 1/n
  vector<size_t> large;
  vector<size_t> small;

  size_t n = prob_.size();

  // Set and allocate memory
  alias_.assign(n, 0);

  // Fill large and small vectors based on 1/n
  for (size_t i = 0; i < n; i++) {
    prob_[i] *= n;
    if (prob_[i] > 1.0) {
      large.push_back(i);
    } else {
      small.push_back(i);
    }
  }

  while (!large.empty() && !small.empty()) {
    int j = small.back();
    int k = large.back();

    // Remove last element of small
    small.pop_back();

    // Update probability and alias based on Vose's algorithm
    prob_[k] += prob_[j] - 1.0;
    alias_[j] = k;

    // Move large index to small vector, if it is no longer large
    if (prob_[k] < 1.0) {
      small.push_back(k);
      large.pop_back();
    }
  }
}

size_t DiscreteIndex::sample(uint64_t* seed) const
{
  // Alias sampling of discrete distribution
  size_t n = prob_.size();
  if (n > 1) {
    size_t u = prn(seed) * n;
    if (prn(seed) < prob_[u]) {
      return u;
    } else {
      return alias_[u];
    }
  } else {
    return 0;
  }
}

void DiscreteIndex::normalize()
{
  // Renormalize density function so that it sums to unity. Note that we save
  // the integral of the distribution so that if it is used as part of another
  // distribution (e.g., Mixture), we know its relative strength.
  integral_ = std::accumulate(prob_.begin(), prob_.end(), 0.0);
  for (auto& p_i : prob_) {
    p_i /= integral_;
  }
}

//==============================================================================
// Discrete implementation
//==============================================================================

Discrete::Discrete(pugi::xml_node node)
{
  auto params = get_node_array<double>(node, "parameters");
  std::size_t n = params.size() / 2;

  // First half is x values, second half is probabilities
  x_.assign(params.begin(), params.begin() + n);
  const double* p = params.data() + n;

  // Store normalized probabilities for cdf()
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    total += p[i];
  p_.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    p_[i] = p[i] / total;

  // Check for bias
  if (check_for_node(node, "bias")) {
    // Get bias probabilities
    auto bias_params = get_node_array<double>(node, "bias");
    if (bias_params.size() != n) {
      openmc::fatal_error(
        "Size mismatch: Attempted to bias Discrete distribution with " +
        std::to_string(n) + " probability entries using a bias with " +
        std::to_string(bias_params.size()) +
        " entries. Please ensure distributions have the same size.");
    }

    // Compute importance weights
    vector<double> p_vec(p, p + n);
    weight_ = compute_importance_weights(p_vec, bias_params);

    // Initialize DiscreteIndex with bias probabilities for sampling
    di_.assign(bias_params);
  } else {
    // Unbiased case: weight_ stays empty
    di_.assign({p, n});
  }
}

Discrete::Discrete(const double* x, const double* p, size_t n) : di_({p, n})
{
  x_.assign(x, x + n);

  // Store normalized probabilities for cdf()
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    total += p[i];
  p_.resize(n);
  for (std::size_t i = 0; i < n; ++i)
    p_[i] = p[i] / total;
}

std::pair<double, double> Discrete::sample(uint64_t* seed) const
{
  size_t idx = di_.sample(seed);
  double wgt = weight_.empty() ? 1.0 : weight_[idx];
  return {x_[idx], wgt};
}

double Discrete::sample_unbiased(uint64_t* seed) const
{
  size_t idx = di_.sample(seed);
  return x_[idx];
}

double Discrete::cdf(double x) const
{
  // Step-function CDF: sum probabilities for all outcomes <= x
  double cum = 0.0;
  for (std::size_t i = 0; i < x_.size(); ++i) {
    if (x_[i] > x)
      break;
    cum += p_[i];
  }
  return cum;
}

double Discrete::quantile(double u) const
{
  // Invert the step-function CDF: find smallest x such that F(x) >= u
  u = std::max(0.0, std::min(1.0, u));
  double cum = 0.0;
  for (std::size_t i = 0; i < x_.size(); ++i) {
    cum += p_[i];
    if (cum >= u)
      return x_[i];
  }
  return x_.back();
}

//==============================================================================
// Uniform implementation
//==============================================================================

Uniform::Uniform(pugi::xml_node node)
{
  auto params = get_node_array<double>(node, "parameters");
  if (params.size() != 2) {
    fatal_error("Uniform distribution must have two "
                "parameters specified.");
  }

  a_ = params.at(0);
  b_ = params.at(1);

  read_bias_from_xml(node);
}

double Uniform::sample_unbiased(uint64_t* seed) const
{
  return a_ + prn(seed) * (b_ - a_);
}

double Uniform::evaluate(double x) const
{
  if (x <= a()) {
    return 0.0;
  } else if (x >= b()) {
    return 0.0;
  } else {
    return 1 / (b() - a());
  }
}

double Uniform::cdf(double x) const
{
  if (x <= a()) {
    return 0.0;
  } else if (x >= b()) {
    return 1.0;
  } else {
    return (x - a()) / (b() - a());
  }
}

double Uniform::quantile(double u) const
{
  // Q(u) = a + u*(b - a)
  u = std::max(0.0, std::min(1.0, u));
  return a_ + u * (b_ - a_);
}

//==============================================================================
// PowerLaw implementation
//==============================================================================

PowerLaw::PowerLaw(pugi::xml_node node)
{
  auto params = get_node_array<double>(node, "parameters");
  if (params.size() != 3) {
    fatal_error("PowerLaw distribution must have three "
                "parameters specified.");
  }

  const double a = params.at(0);
  const double b = params.at(1);
  const double n = params.at(2);

  offset_ = std::pow(a, n + 1);
  span_ = std::pow(b, n + 1) - offset_;
  ninv_ = 1 / (n + 1);

  read_bias_from_xml(node);
}

double PowerLaw::evaluate(double x) const
{
  if (x <= a()) {
    return 0.0;
  } else if (x >= b()) {
    return 0.0;
  } else {
    int pwr = n() + 1;
    double norm = pwr / span_;
    return norm * std::pow(std::fabs(x), n());
  }
}

double PowerLaw::cdf(double x) const
{
  if (x <= a()) {
    return 0.0;
  } else if (x >= b()) {
    return 1.0;
  } else {
    // CDF is (x^(n+1) - a^(n+1)) / (b^(n+1) - a^(n+1))
    return (std::pow(x, n() + 1) - offset_) / span_;
  }
}

double PowerLaw::quantile(double u) const
{
  // Q(u) = (a^(n+1) + u * (b^(n+1) - a^(n+1)))^(1/(n+1))
  u = std::max(0.0, std::min(1.0, u));
  return std::pow(offset_ + u * span_, ninv_);
}

double PowerLaw::sample_unbiased(uint64_t* seed) const
{
  return std::pow(offset_ + prn(seed) * span_, ninv_);
}

//==============================================================================
// Maxwell implementation
//==============================================================================

Maxwell::Maxwell(pugi::xml_node node)
{
  theta_ = std::stod(get_node_value(node, "parameters"));

  read_bias_from_xml(node);
}

double Maxwell::sample_unbiased(uint64_t* seed) const
{
  return maxwell_spectrum(theta_, seed);
}

double Maxwell::evaluate(double x) const
{
  double c = (2.0 / SQRT_PI) * std::pow(theta_, -1.5);
  return c * std::sqrt(x) * std::exp(-x / theta_);
}

double Maxwell::cdf(double x) const
{
  // Maxwell CDF using the regularized lower incomplete gamma function:
  // F(x) = gamma(3/2, x/theta) / Gamma(3/2)
  // which equals erf(sqrt(x/theta)) - 2*sqrt(x/(pi*theta)) * exp(-x/theta)
  if (x <= 0.0)
    return 0.0;
  double t = std::sqrt(x / theta_);
  return std::erf(t) - (2.0 / SQRT_PI) * t * std::exp(-t * t);
}

double Maxwell::quantile(double u) const
{
  // Use lazy-built quantile table
  std::call_once(quantile_init_, [this]() { build_quantile_table(); });
  return quantile_table_(u);
}

void Maxwell::build_quantile_table() const
{
  auto pdf = [this](double x) { return this->evaluate(x); };
  double x_hi = find_upper_bound(pdf, 0.0, 10.0 * theta_, 1e-14);
  quantile_table_ = openmc::build_quantile_table(pdf, 0.0, x_hi);
}

//==============================================================================
// Watt implementation
//==============================================================================

Watt::Watt(pugi::xml_node node)
{
  auto params = get_node_array<double>(node, "parameters");
  if (params.size() != 2)
    openmc::fatal_error("Watt energy distribution must have two "
                        "parameters specified.");

  a_ = params.at(0);
  b_ = params.at(1);

  read_bias_from_xml(node);
}

double Watt::sample_unbiased(uint64_t* seed) const
{
  return watt_spectrum(a_, b_, seed);
}

double Watt::evaluate(double x) const
{
  double c =
    2.0 / (std::sqrt(PI * b_) * std::pow(a_, 1.5) * std::exp(a_ * b_ / 4.0));
  return c * std::exp(-x / a_) * std::sinh(std::sqrt(b_ * x));
}

double Watt::cdf(double x) const
{
  // Numerical integration of the Watt PDF using the trapezoidal rule.
  // The Watt spectrum has no closed-form CDF.
  if (x <= 0.0)
    return 0.0;

  // Use 500 trapezoid panels for accuracy
  constexpr int n_panels = 500;
  double dx = x / n_panels;
  double sum = 0.0;
  double f_prev = evaluate(0.0);
  for (int i = 1; i <= n_panels; ++i) {
    double xi = i * dx;
    double f_curr = evaluate(xi);
    sum += 0.5 * (f_prev + f_curr) * dx;
    f_prev = f_curr;
  }
  return std::min(sum, 1.0);
}

double Watt::quantile(double u) const
{
  // Use lazy-built quantile table
  std::call_once(quantile_init_, [this]() { build_quantile_table(); });
  return quantile_table_(u);
}

void Watt::build_quantile_table() const
{
  auto pdf = [this](double x) { return this->evaluate(x); };
  double x_hi = find_upper_bound(pdf, 0.0, 10.0 * a_, 1e-14);
  quantile_table_ = openmc::build_quantile_table(pdf, 0.0, x_hi);
}

//==============================================================================
// Normal implementation
//==============================================================================

Normal::Normal(double mean_value, double std_dev, double lower, double upper)
  : mean_value_ {mean_value}, std_dev_ {std_dev}, lower_ {lower}, upper_ {upper}
{
  compute_normalization();
}

Normal::Normal(pugi::xml_node node)
{
  auto params = get_node_array<double>(node, "parameters");
  if (params.size() != 2 && params.size() != 4) {
    openmc::fatal_error("Normal energy distribution must have two "
                        "parameters (mean, std_dev) or four parameters "
                        "(mean, std_dev, lower, upper) specified.");
  }

  mean_value_ = params.at(0);
  std_dev_ = params.at(1);

  // Optional truncation bounds
  if (params.size() == 4) {
    lower_ = params.at(2);
    upper_ = params.at(3);
  } else {
    lower_ = -INFTY;
    upper_ = INFTY;
  }

  compute_normalization();
  read_bias_from_xml(node);
}

void Normal::compute_normalization()
{
  // Validate bounds
  if (lower_ >= upper_) {
    openmc::fatal_error(
      "Normal distribution lower bound must be less than upper bound.");
  }

  // Check if truncation bounds are finite
  is_truncated_ = (lower_ > -INFTY || upper_ < INFTY);

  if (is_truncated_) {
    double alpha = (lower_ - mean_value_) / std_dev_;
    double beta = (upper_ - mean_value_) / std_dev_;
    double cdf_diff = standard_normal_cdf(beta) - standard_normal_cdf(alpha);

    if (cdf_diff <= 0.0) {
      openmc::fatal_error(
        "Normal distribution truncation bounds exclude entire distribution.");
    }
    norm_factor_ = 1.0 / cdf_diff;
  } else {
    norm_factor_ = 1.0;
  }
}

double Normal::sample_unbiased(uint64_t* seed) const
{
  if (!is_truncated_) {
    return normal_variate(mean_value_, std_dev_, seed);
  }

  // Rejection sampling for truncated normal
  double x;
  do {
    x = normal_variate(mean_value_, std_dev_, seed);
  } while (x < lower_ || x > upper_);
  return x;
}

double Normal::evaluate(double x) const
{
  // Return 0 outside truncation bounds
  if (x < lower_ || x > upper_) {
    return 0.0;
  }

  // Standard normal PDF value
  double pdf = (1.0 / (std::sqrt(2.0 * PI) * std_dev_)) *
               std::exp(-std::pow((x - mean_value_), 2.0) /
                        (2.0 * std::pow(std_dev_, 2.0)));

  // Apply normalization for truncation
  return pdf * norm_factor_;
}

double Normal::cdf(double x) const
{
  // Return 0/1 outside truncation bounds
  if (x <= lower_)
    return 0.0;
  if (x >= upper_)
    return 1.0;

  // Standardize
  double z = (x - mean_value_) / std_dev_;
  double F = standard_normal_cdf(z);

  if (!is_truncated_) {
    return F;
  }

  // For truncated normal: F_trunc(x) = (Phi(z) - Phi(alpha)) / (Phi(beta) - Phi(alpha))
  double alpha = (lower_ - mean_value_) / std_dev_;
  double beta = (upper_ - mean_value_) / std_dev_;
  double F_alpha = standard_normal_cdf(alpha);
  double F_beta = standard_normal_cdf(beta);
  return (F - F_alpha) / (F_beta - F_alpha);
}

double Normal::quantile(double u) const
{
  u = std::max(0.0, std::min(1.0, u));

  if (!is_truncated_) {
    // Q(u) = mu + sigma * Phi^{-1}(u)
    return mean_value_ + std_dev_ * normal_percentile(u);
  }

  // For truncated normal:
  // Q(u) = mu + sigma * Phi^{-1}(Phi(alpha) + u * (Phi(beta) - Phi(alpha)))
  double alpha = (lower_ - mean_value_) / std_dev_;
  double beta = (upper_ - mean_value_) / std_dev_;
  double F_alpha = standard_normal_cdf(alpha);
  double F_beta = standard_normal_cdf(beta);
  double p = F_alpha + u * (F_beta - F_alpha);
  // Clamp to avoid numerical issues at boundaries
  p = std::max(1e-16, std::min(1.0 - 1e-16, p));
  return mean_value_ + std_dev_ * normal_percentile(p);
}

//==============================================================================
// Tabular implementation
//==============================================================================

Tabular::Tabular(pugi::xml_node node)
{
  if (check_for_node(node, "interpolation")) {
    std::string temp = get_node_value(node, "interpolation");
    if (temp == "histogram") {
      interp_ = Interpolation::histogram;
    } else if (temp == "linear-linear") {
      interp_ = Interpolation::lin_lin;
    } else if (temp == "log-linear") {
      interp_ = Interpolation::log_lin;
    } else if (temp == "log-log") {
      interp_ = Interpolation::log_log;
    } else {
      openmc::fatal_error(
        "Unsupported interpolation type for distribution: " + temp);
    }
  } else {
    interp_ = Interpolation::histogram;
  }

  // Read and initialize tabular distribution. If number of parameters is odd,
  // add an extra zero for the 'p' array.
  auto params = get_node_array<double>(node, "parameters");
  if (params.size() % 2 != 0) {
    params.push_back(0.0);
  }
  std::size_t n = params.size() / 2;
  const double* x = params.data();
  const double* p = x + n;
  init(x, p, n);

  read_bias_from_xml(node);
}

Tabular::Tabular(const double* x, const double* p, int n, Interpolation interp,
  const double* c)
  : interp_ {interp}
{
  init(x, p, n, c);
}

void Tabular::init(
  const double* x, const double* p, std::size_t n, const double* c)
{
  // Copy x/p arrays into vectors
  std::copy(x, x + n, std::back_inserter(x_));
  std::copy(p, p + n, std::back_inserter(p_));

  // Calculate cumulative distribution function
  if (c) {
    std::copy(c, c + n, std::back_inserter(c_));
  } else {
    c_.resize(n);
    c_[0] = 0.0;
    for (int i = 1; i < n; ++i) {
      if (interp_ == Interpolation::histogram) {
        c_[i] = c_[i - 1] + p_[i - 1] * (x_[i] - x_[i - 1]);
      } else if (interp_ == Interpolation::lin_lin) {
        c_[i] = c_[i - 1] + 0.5 * (p_[i - 1] + p_[i]) * (x_[i] - x_[i - 1]);
      } else if (interp_ == Interpolation::log_lin) {
        double m = std::log(p_[i] / p_[i - 1]) / (x_[i] - x_[i - 1]);
        c_[i] = c_[i - 1] + p_[i - 1] * (x_[i] - x_[i - 1]) *
                              exprel(m * (x_[i] - x_[i - 1]));
      } else if (interp_ == Interpolation::log_log) {
        double m = std::log((x_[i] * p_[i]) / (x_[i - 1] * p_[i - 1])) /
                   std::log(x_[i] / x_[i - 1]);
        c_[i] = c_[i - 1] + x_[i - 1] * p_[i - 1] *
                              std::log(x_[i] / x_[i - 1]) *
                              exprel(m * std::log(x_[i] / x_[i - 1]));
      } else {
        UNREACHABLE();
      }
    }
  }

  // Normalize density and distribution functions. Note that we save the
  // integral of the distribution so that if it is used as part of another
  // distribution (e.g., Mixture), we know its relative strength.
  integral_ = c_[n - 1];
  for (int i = 0; i < n; ++i) {
    p_[i] = p_[i] / integral_;
    c_[i] = c_[i] / integral_;
  }
}

double Tabular::sample_unbiased(uint64_t* seed) const
{
  // Sample value of CDF
  double c = prn(seed);

  // Find first CDF bin which is above the sampled value
  double c_i = c_[0];
  int i;
  std::size_t n = c_.size();
  for (i = 0; i < n - 1; ++i) {
    if (c <= c_[i + 1])
      break;
    c_i = c_[i + 1];
  }

  // Determine bounding PDF values
  double x_i = x_[i];
  double p_i = p_[i];

  if (interp_ == Interpolation::histogram) {
    // Histogram interpolation
    if (p_i > 0.0) {
      return x_i + (c - c_i) / p_i;
    } else {
      return x_i;
    }
  } else if (interp_ == Interpolation::lin_lin) {
    // Linear-linear interpolation
    double x_i1 = x_[i + 1];
    double p_i1 = p_[i + 1];

    double m = (p_i1 - p_i) / (x_i1 - x_i);
    if (m == 0.0) {
      return x_i + (c - c_i) / p_i;
    } else {
      return x_i +
             (std::sqrt(std::max(0.0, p_i * p_i + 2 * m * (c - c_i))) - p_i) /
               m;
    }
  } else if (interp_ == Interpolation::log_lin) {
    // Log-linear interpolation
    double x_i1 = x_[i + 1];
    double p_i1 = p_[i + 1];

    double m = std::log(p_i1 / p_i) / (x_i1 - x_i);
    double f = (c - c_i) / p_i;
    return x_i + f * log1prel(m * f);
  } else if (interp_ == Interpolation::log_log) {
    // Log-Log interpolation
    double x_i1 = x_[i + 1];
    double p_i1 = p_[i + 1];

    double m = std::log((x_i1 * p_i1) / (x_i * p_i)) / std::log(x_i1 / x_i);
    double f = (c - c_i) / (p_i * x_i);
    return x_i * std::exp(f * log1prel(m * f));
  } else {
    UNREACHABLE();
  }
}

double Tabular::evaluate(double x) const
{
  int i;

  if (interp_ == Interpolation::histogram) {
    i = std::upper_bound(x_.begin(), x_.end(), x) - x_.begin() - 1;
    if (i < 0 || i >= static_cast<int>(p_.size())) {
      return 0.0;
    } else {
      return p_[i];
    }
  } else {
    i = std::lower_bound(x_.begin(), x_.end(), x) - x_.begin() - 1;

    if (i < 0 || i >= static_cast<int>(p_.size()) - 1) {
      return 0.0;
    } else {
      double x0 = x_[i];
      double x1 = x_[i + 1];
      double p0 = p_[i];
      double p1 = p_[i + 1];

      double t = (x - x0) / (x1 - x0);
      return (1 - t) * p0 + t * p1;
    }
  }
}

double Tabular::cdf(double x) const
{
  // Return 0/1 for values outside the tabulated range
  if (x <= x_.front())
    return 0.0;
  if (x >= x_.back())
    return 1.0;

  // Find the bin containing x
  auto it = std::upper_bound(x_.begin(), x_.end(), x);
  int i = static_cast<int>(it - x_.begin()) - 1;
  i = std::max(0, std::min(i, static_cast<int>(x_.size()) - 2));

  // Interpolate within the bin to get CDF value
  if (interp_ == Interpolation::histogram) {
    return c_[i] + p_[i] * (x - x_[i]);
  } else if (interp_ == Interpolation::lin_lin) {
    double dx = x_[i + 1] - x_[i];
    double t = (x - x_[i]) / dx;
    double p_at_x = (1.0 - t) * p_[i] + t * p_[i + 1];
    return c_[i] + 0.5 * (p_[i] + p_at_x) * (x - x_[i]);
  } else if (interp_ == Interpolation::log_lin) {
    double m = std::log(p_[i + 1] / p_[i]) / (x_[i + 1] - x_[i]);
    double dx = x - x_[i];
    return c_[i] + p_[i] * dx * exprel(m * dx);
  } else if (interp_ == Interpolation::log_log) {
    double m = std::log((x_[i + 1] * p_[i + 1]) / (x_[i] * p_[i])) /
               std::log(x_[i + 1] / x_[i]);
    double lnr = std::log(x / x_[i]);
    return c_[i] + x_[i] * p_[i] * lnr * exprel(m * lnr);
  } else {
    UNREACHABLE();
  }
}

double Tabular::quantile(double u) const
{
  // This is the same algorithm as sample_unbiased(), but takes a CDF value
  // directly instead of drawing one from prn(seed). This IS the inverse CDF.
  u = std::max(0.0, std::min(1.0, u));

  // Find first CDF bin which is above the sampled value
  double c_i = c_[0];
  int i;
  std::size_t n = c_.size();
  for (i = 0; i < static_cast<int>(n) - 1; ++i) {
    if (u <= c_[i + 1])
      break;
    c_i = c_[i + 1];
  }

  // Determine bounding values
  double x_i = x_[i];
  double p_i = p_[i];

  if (interp_ == Interpolation::histogram) {
    if (p_i > 0.0) {
      return x_i + (u - c_i) / p_i;
    } else {
      return x_i;
    }
  } else if (interp_ == Interpolation::lin_lin) {
    double x_i1 = x_[i + 1];
    double p_i1 = p_[i + 1];
    double m = (p_i1 - p_i) / (x_i1 - x_i);
    if (m == 0.0) {
      return x_i + (u - c_i) / p_i;
    } else {
      return x_i +
             (std::sqrt(std::max(0.0, p_i * p_i + 2 * m * (u - c_i))) - p_i) /
               m;
    }
  } else if (interp_ == Interpolation::log_lin) {
    double x_i1 = x_[i + 1];
    double p_i1 = p_[i + 1];
    double m = std::log(p_i1 / p_i) / (x_i1 - x_i);
    double f = (u - c_i) / p_i;
    return x_i + f * log1prel(m * f);
  } else if (interp_ == Interpolation::log_log) {
    double x_i1 = x_[i + 1];
    double p_i1 = p_[i + 1];
    double m = std::log((x_i1 * p_i1) / (x_i * p_i)) / std::log(x_i1 / x_i);
    double f = (u - c_i) / (p_i * x_i);
    return x_i * std::exp(f * log1prel(m * f));
  } else {
    UNREACHABLE();
  }
}

//==============================================================================
// Equiprobable implementation
//==============================================================================

double Equiprobable::sample_unbiased(uint64_t* seed) const
{
  std::size_t n = x_.size();

  double r = prn(seed);
  int i = std::floor((n - 1) * r);

  double xl = x_[i];
  double xr = x_[i + i];
  return xl + ((n - 1) * r - i) * (xr - xl);
}

double Equiprobable::evaluate(double x) const
{
  double x_min = *std::min_element(x_.begin(), x_.end());
  double x_max = *std::max_element(x_.begin(), x_.end());

  if (x < x_min || x > x_max) {
    return 0.0;
  } else {
    return 1.0 / (x_max - x_min);
  }
}

double Equiprobable::cdf(double x) const
{
  std::size_t n = x_.size();
  if (n == 0)
    return 0.0;

  if (x <= x_.front())
    return 0.0;
  if (x >= x_.back())
    return 1.0;

  // Each of the (n-1) bins has equal probability 1/(n-1).
  // Find which bin x falls in and interpolate linearly.
  double bin_prob = 1.0 / (n - 1);
  for (std::size_t i = 0; i < n - 1; ++i) {
    if (x <= x_[i + 1]) {
      double frac = (x - x_[i]) / (x_[i + 1] - x_[i]);
      return (i + frac) * bin_prob;
    }
  }
  return 1.0;
}

double Equiprobable::quantile(double u) const
{
  // Equiprobable: each of (n-1) bins has probability 1/(n-1).
  // Same logic as sample_unbiased but with explicit u instead of prn(seed).
  u = std::max(0.0, std::min(1.0, u));
  std::size_t n = x_.size();
  int i = static_cast<int>(std::floor((n - 1) * u));
  i = std::min(i, static_cast<int>(n) - 2);
  double xl = x_[i];
  double xr = x_[i + 1];
  return xl + ((n - 1) * u - i) * (xr - xl);
}

//==============================================================================
// Mixture implementation
//==============================================================================

Mixture::Mixture(pugi::xml_node node)
{
  vector<double> probabilities;

  // First pass: collect distributions and their probabilities
  for (pugi::xml_node pair : node.children("pair")) {
    // Check that required data exists
    if (!pair.attribute("probability"))
      fatal_error("Mixture pair element does not have probability.");
    if (!pair.child("dist"))
      fatal_error("Mixture pair element does not have a distribution.");

    // Get probability and distribution
    double p = std::stod(pair.attribute("probability").value());
    auto dist = distribution_from_xml(pair.child("dist"));

    // Weight probability by the distribution's integral
    double weighted_prob = p * dist->integral();
    probabilities.push_back(weighted_prob);
    distribution_.push_back(std::move(dist));
  }

  // Save sum of weighted probabilities
  integral_ = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);

  // Store normalized mixture probabilities for evaluate()
  std::size_t n = probabilities.size();
  prob_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    prob_[i] = probabilities[i] / integral_;
  }

  // Check for bias
  if (check_for_node(node, "bias")) {
    // Get bias probabilities
    auto bias_params = get_node_array<double>(node, "bias");
    if (bias_params.size() != n) {
      openmc::fatal_error(
        "Size mismatch: Attempted to bias Mixture distribution with " +
        std::to_string(n) + " components using a bias with " +
        std::to_string(bias_params.size()) +
        " entries. Please ensure distributions have the same size.");
    }

    // Compute importance weights
    weight_ = compute_importance_weights(probabilities, bias_params);

    // Initialize DiscreteIndex with bias probabilities for sampling
    di_.assign(bias_params);
  } else {
    // Unbiased case: weight_ stays empty
    di_.assign(probabilities);
  }
}

std::pair<double, double> Mixture::sample(uint64_t* seed) const
{
  size_t idx = di_.sample(seed);

  // Sample the chosen distribution
  auto [val, sub_wgt] = distribution_[idx]->sample(seed);

  // Multiply by component selection weight
  double mix_wgt = weight_.empty() ? 1.0 : weight_[idx];
  return {val, mix_wgt * sub_wgt};
}

double Mixture::sample_unbiased(uint64_t* seed) const
{
  size_t idx = di_.sample(seed);
  return distribution_[idx]->sample(seed).first;
}

double Mixture::evaluate(double x) const
{
  // Mixture PDF is the probability-weighted sum of component PDFs
  double result = 0.0;
  for (std::size_t i = 0; i < distribution_.size(); ++i) {
    result += prob_[i] * distribution_[i]->evaluate(x);
  }
  return result;
}

double Mixture::cdf(double x) const
{
  // Mixture CDF is the probability-weighted sum of component CDFs
  double result = 0.0;
  for (std::size_t i = 0; i < distribution_.size(); ++i) {
    result += prob_[i] * distribution_[i]->cdf(x);
  }
  return result;
}

double Mixture::quantile(double u) const
{
  // Use lazy-built quantile table (must invert the full mixture CDF,
  // NOT sum component quantiles)
  std::call_once(quantile_init_, [this]() { build_quantile_table(); });
  return quantile_table_(u);
}

void Mixture::build_quantile_table() const
{
  auto pdf = [this](double x) { return this->evaluate(x); };

  // Determine the effective domain from all components.
  // Use CDF to find bounds where mixture CDF ≈ 0 and ≈ 1.
  // Start by finding the range where PDF is non-negligible.
  double x_lo = 0.0;
  double x_hi = find_upper_bound(pdf, 0.0, 1.0, 1e-14);

  // Also search for negative support (e.g., Normal components)
  // March leftward from 0 to find where PDF becomes negligible
  double f_peak = 0.0;
  for (int i = 0; i <= 50; ++i) {
    double xi = x_lo + (x_hi - x_lo) * i / 50.0;
    f_peak = std::max(f_peak, pdf(xi));
  }
  if (f_peak > 0.0) {
    double threshold = f_peak * 1e-14;
    double step = std::max(1.0, x_hi - x_lo);
    double x_test = 0.0;
    for (int iter = 0; iter < 50; ++iter) {
      x_test -= step;
      if (pdf(x_test) < threshold) {
        x_lo = x_test;
        break;
      }
      step *= 2.0;
    }
  }

  quantile_table_ = openmc::build_quantile_table(pdf, x_lo, x_hi);
}

//==============================================================================
// Helper function
//==============================================================================

UPtrDist distribution_from_xml(pugi::xml_node node)
{
  if (!check_for_node(node, "type"))
    openmc::fatal_error("Distribution type must be specified.");

  // Determine type of distribution
  std::string type = get_node_value(node, "type", true, true);

  // Allocate extension of Distribution
  UPtrDist dist;
  if (type == "uniform") {
    dist = UPtrDist {new Uniform(node)};
  } else if (type == "powerlaw") {
    dist = UPtrDist {new PowerLaw(node)};
  } else if (type == "maxwell") {
    dist = UPtrDist {new Maxwell(node)};
  } else if (type == "watt") {
    dist = UPtrDist {new Watt(node)};
  } else if (type == "normal") {
    dist = UPtrDist {new Normal(node)};
  } else if (type == "discrete") {
    dist = UPtrDist {new Discrete(node)};
  } else if (type == "tabular") {
    dist = UPtrDist {new Tabular(node)};
  } else if (type == "mixture") {
    dist = UPtrDist {new Mixture(node)};
  } else if (type == "muir") {
    openmc::fatal_error(
      "'muir' distributions are now specified using the openmc.stats.muir() "
      "function in Python. Please regenerate your XML files.");
  } else {
    openmc::fatal_error("Invalid distribution type: " + type);
  }
  return dist;
}

} // namespace openmc
