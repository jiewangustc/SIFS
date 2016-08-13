#ifndef _SSVM_SIFS_HPP_
#define _SSVM_SIFS_HPP_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sdm/lib/io.hpp>
#include <random>
#include <numeric>
#include <vector>
#include <iterator>
#include <chrono>
#include <unordered_set>

using SpMatRd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpMatCd = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using scm_iit = Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator;
using srm_iit = Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;

using sec = std::chrono::seconds;
using mil_sec = std::chrono::milliseconds;
using sys_clk = std::chrono::system_clock;

using u_set = std::unordered_set<int>;

/**
 * elastic-net regularized smoothed sparse svm with SIFS
 *   min .5 * alpha * |w|^2 + beta * |w|_1 + (1/n) * sum_{i=1}^n l(1 - bar{x}_i^T w)
 *   where l(t) = 0, if t < 0
 *              = t^2 / (2 * gamma), if 0 <= t <= gamma
 *              = t - gamma/2, if t > gamma
 *         \bar{x}_i = y_i * x_i
 *
 * The dual problem is 
 *   max (1/n) * 1^T*theta - gamma/(2n) * |theta|^2 - 1/(2alpha) * |S_beta(\bar{X}^T*theta/n)|^2
 *   s.t. 0 <= theta_i <= 1, \bar{X}^T = [\bar{x}_1, \ldots, \bar{x}_n]
 *
 * This implementation is adapted from s3fs (https://github.com/husk214/s3fs)
 * Shibagaki, Atsushi, et al. Simultaneous Safe Screening of Features and Samples in Doubly Sparse Modeling (ICML'16)
 *
 */

class ssvm_sifs {
  public:
    ssvm_sifs(const std::string& input_fn,
         const double& alpha, const double& beta,
         const double& gamma = 0.5, const double& tol = 1e-8,
         const int& max_iter = 10000, const int& chk_fre = 1,
         const int& scr_max_iter = 0);
    ~ssvm_sifs();

    int get_n_sams(void) const;
    int get_n_feas(void) const;

    double get_primal_obj(void) const;
    double get_dual_obj(void) const; 
    double get_duality_gap(void) const; 

    Eigen::VectorXd get_dual_sol(void) const;
    Eigen::VectorXd get_primal_sol(void) const;
    
    void set_alpha(const double& alpha, const bool& ws = true);
    void set_beta(const double& beta);
    void set_stop_tol(const double& tol);
    
    void compute_primal_obj(const bool& flag_comp_loss); // compute primal objective
    void compute_dual_obj(const bool& flag_comp_XTdsol); // compute dual objective
    void compute_duality_gap(const bool& flag_comp_loss,
                             const bool& flag_comp_XTdsol); // compute duality gap
 
    void update_psol(const bool& flag_comp_XTdsol);

    void train(void);
    void train_sifs(const int& scr_option = 0);

    void sample_screening(void);
    void feature_screening(void);
 
    void sifs(const bool& sample_scr_first = true);
    void ifs(void);
    void iss(void);

    void clear_idx(void);

    int get_n_L(void) const;
    int get_n_R(void) const;
    int get_n_F(void) const;
    int get_iter(void) const;

    double get_alpha_max(void) const;
    double get_beta_max(void) const;

    SpMatRd X_;  // \bar{X}, each row contains one sample
    SpMatCd X_CM_;  // column major representation of \bar{X}
    Eigen::ArrayXd y_;  // training labels
    
    double fea_scr_time_;
    double sam_scr_time_;
    double scr_time_;

  protected:
    int n_sams_;
    int n_feas_;

    double alpha_;
    double beta_;
    double gamma_;
    double tol_;
    int max_iter_;
    int chk_fre_;
    int iter_;
 
    double alpha_max_;
    double beta_max_;
    int scr_max_iter_;

    double pobj_;
    double dobj_;
    double duality_gap_;
    double loss_;

    double inv_n_sams_;
    double inv_alpha_;
    double inv_gamma_;

    Eigen::ArrayXd one_over_XTones_;  // \bar{X}^T * ones / n
    Eigen::VectorXd psol_;
    Eigen::VectorXd dsol_;
    Eigen::VectorXd XTdsol_; // \bar{X}^T * theta / n
    Eigen::ArrayXd Xw_comp_; // 1 - \bar{X} * w
    
    Eigen::ArrayXd Xi_norm_;
    Eigen::ArrayXd Xi_norm_sq_;
    Eigen::ArrayXd Xj_norm_;
    Eigen::ArrayXd Xj_norm_sq_;
    
    std::vector<int> all_ins_index_;
    std::vector<int> all_fea_index_;
    u_set idx_nsv_L_;
    u_set idx_nsv_R_;
    u_set idx_Dc_;
    u_set idx_F_;
    u_set idx_Fc_;
    std::vector<int> idx_Dc_vec_;
    std::vector<int> idx_Fc_vec_;
    Eigen::VectorXd idx_Dc_flag_;
    Eigen::VectorXd idx_Fc_flag_;
    
    double ref_alpha_; // alpha_0
    double dif_alpha_ratio_; // (alpha - alpha_0) / alpha
    double sum_alpha_ratio_; // (alpha + alpha_0) / alpha
    Eigen::VectorXd ref_psol_;  // reference primal solution w_0,
                                //also center of primal optimum estimation
    Eigen::VectorXd ref_dsol_;
    Eigen::VectorXd approx_dsol_c_;  // center of dual optimum estimation
    double approx_dsol_r_sq_;  // square of radius of dual optimum estimation
    double approx_dsol_r_L_sq_;
    double approx_dsol_r_R_sq_;
    double approx_psol_r_sq_;  // |w_0|^2
    double approx_psol_r_F_sq_;  // |w_0(F)|^2

};

template <typename _Tp> inline _Tp val_sign(_Tp val) {
    return 1.0 - (val <= 0.0) - (val < 0.0);
}

#endif
