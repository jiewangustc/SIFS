#include "ssvm_sifs.hpp"

ssvm_sifs::ssvm_sifs(const std::string& input_fn,
                     const double& alpha, const double& beta, const double& gamma,
                     const double& tol, const int& max_iter, const int& chk_fre,
                     const int& scr_max_iter)
    : alpha_(alpha), beta_(beta), gamma_(gamma),
      tol_(tol), max_iter_(max_iter), chk_fre_(chk_fre),
      scr_max_iter_(scr_max_iter), pobj_(0.0), dobj_(0.0) {

    sdm::load_libsvm_binary(X_, y_, input_fn);
    for (int i = 0; i < X_.rows(); ++i){
        X_.row(i) *= y_[i];
    }
    X_CM_ = X_;
    n_sams_ = X_.rows();
    n_feas_ = X_.cols();

    inv_n_sams_ = 1.0 / static_cast<double>(n_sams_);
    inv_gamma_ = 1.0 / gamma_;
    inv_alpha_ = 1.0 / alpha_;

    one_over_XTones_ =
        inv_n_sams_ * (X_CM_.transpose() * Eigen::VectorXd::Ones(n_sams_)).array();
    // if beta > beta_max, we always have naive solutions
    beta_max_ = one_over_XTones_.abs().maxCoeff();
    std::cout << "beta_max = " << beta_max_ << std::endl;
    if (beta_ >= beta_max_) {
        std::cout << "beta >= beta_max: always have naive solutions" << std::endl;
    }

    Eigen::ArrayXd temp =
        (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
        (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
    alpha_max_ = (1.0/(1.0 - gamma_)) * (X_ * temp.matrix()).maxCoeff();
    // if alpha > alpha_max, we have closed form solutions

    if (alpha_max_ > 0) {
        ref_alpha_ = alpha_max_;
        ref_dsol_ = Eigen::VectorXd::Ones(n_sams_);
        ref_psol_ = temp * (1.0 / ref_alpha_);
        dif_alpha_ratio_ = 0.5 * (alpha_ - ref_alpha_) / alpha_;
        sum_alpha_ratio_ = 0.5 * (alpha_ + ref_alpha_) / alpha_;
    } else{
        ref_alpha_ = 0;
        ref_dsol_ = Eigen::VectorXd::Ones(n_sams_);
        ref_psol_ = Eigen::VectorXd::Zero(n_feas_);
        dif_alpha_ratio_ = 0.5;
        sum_alpha_ratio_ = 0.5;
    } 

    dsol_ = Eigen::VectorXd::Ones(n_sams_);
    Xw_comp_ = dsol_;
    psol_ = Eigen::VectorXd::Zero(n_feas_);
    XTdsol_ = Eigen::VectorXd::Zero(n_feas_);

    Xi_norm_.resize(n_sams_);
    for (int i = 0; i < n_sams_; ++i) {
        Xi_norm_[i] = X_.row(i).norm();
    }

    Xj_norm_.resize(n_feas_);
    for (int j = 0; j < n_feas_; ++j) {
        Xj_norm_[j] = X_CM_.col(j).norm();
    }

    Xi_norm_sq_ = Xi_norm_.square();
    Xj_norm_sq_ = Xj_norm_.square();

    all_ins_index_.resize(n_sams_);
    std::iota(std::begin(all_ins_index_), std::end(all_ins_index_), 0);
    all_fea_index_.resize(n_feas_);
    std::iota(std::begin(all_fea_index_), std::end(all_fea_index_), 0);

    idx_Fc_flag_ = Eigen::VectorXd::Ones(n_feas_);
    idx_Dc_flag_ = Eigen::VectorXd::Ones(n_sams_);

    idx_Dc_.clear();
    idx_Fc_.clear();
    idx_Dc_.insert(std::begin(all_ins_index_), std::end(all_ins_index_));
    idx_Fc_.insert(std::begin(all_fea_index_), std::end(all_fea_index_));

    idx_nsv_L_.clear();
    idx_nsv_R_.clear();
    idx_F_.clear();

    std::copy(std::begin(all_ins_index_), std::end(all_ins_index_),
              std::back_inserter(idx_Dc_vec_));
    std::copy(std::begin(all_fea_index_), std::end(all_fea_index_),
              std::back_inserter(idx_Fc_vec_));

    approx_dsol_r_L_sq_ = 0.0;
    approx_dsol_r_R_sq_ = 0.0;
    approx_psol_r_F_sq_ = 0.0;
    
    fea_scr_time_ = 0.0;
    sam_scr_time_ = 0.0;
    scr_time_ = 0.0;

    iter_ = 0;
    duality_gap_ = std::numeric_limits<double>::max();
}

ssvm_sifs::~ssvm_sifs() {}

int ssvm_sifs::get_n_sams(void) const { return n_sams_; }
int ssvm_sifs::get_n_feas(void) const { return n_feas_; }

double ssvm_sifs::get_primal_obj(void) const { return pobj_; }
double ssvm_sifs::get_dual_obj(void) const { return dobj_; }
double ssvm_sifs::get_duality_gap(void) const { return duality_gap_; }

int ssvm_sifs::get_n_L(void) const { return idx_nsv_L_.size(); }
int ssvm_sifs::get_n_R(void) const { return idx_nsv_R_.size(); }
int ssvm_sifs::get_n_F(void) const { return idx_F_.size(); }

int ssvm_sifs::get_iter(void) const { return iter_; }

double ssvm_sifs::get_alpha_max(void) const { return alpha_max_; }
double ssvm_sifs::get_beta_max(void) const { return beta_max_; }

Eigen::VectorXd ssvm_sifs::get_dual_sol(void) const { return dsol_; }
Eigen::VectorXd ssvm_sifs::get_primal_sol(void) const { return psol_; }

void ssvm_sifs::set_stop_tol(const double& tol) { tol_ = tol; }

void ssvm_sifs::compute_primal_obj(const bool& flag_comp_loss) {
    pobj_ = (.5 * alpha_) * psol_.squaredNorm() + (beta_) * psol_.lpNorm<1>();

    loss_ = 0.0;
    if (flag_comp_loss){
        Xw_comp_ = 1 - (X_ * psol_).array();
        double Xw_comp_i;
        for (int i = 0; i < n_sams_; ++i) {
            Xw_comp_i = Xw_comp_[i];
            if (Xw_comp_i > gamma_){
                loss_ += Xw_comp_i - .5 * gamma_;
            } else if (Xw_comp_i > 0.0) {
                loss_ += .5 * Xw_comp_i * Xw_comp_i * inv_gamma_;
            }
        }
    }
    pobj_ = pobj_ + loss_ * inv_n_sams_;
}

void ssvm_sifs::compute_dual_obj(const bool& flag_comp_XTdsol) {
    if (flag_comp_XTdsol){
        update_psol(true);
    }
    Eigen::ArrayXd temp = XTdsol_.array().abs() - beta_;
    dobj_ =
        (dsol_.sum() - (.5 * gamma_) * dsol_.squaredNorm()) * inv_n_sams_
        - (0.5 * inv_alpha_) * (temp >= 0.0).select(temp, 0.0).square().sum();
}

void ssvm_sifs::compute_duality_gap(const bool& flag_comp_loss,
                                    const bool& flag_comp_XTdsol) {
    compute_primal_obj(flag_comp_loss);
    compute_dual_obj(flag_comp_XTdsol);

    duality_gap_ = std::max(0.0, pobj_ - dobj_);
}

void ssvm_sifs::update_psol(const bool& flag_comp_XTdsol) {
    if (flag_comp_XTdsol) {
        XTdsol_.setZero();
        for (int i = 0; i < n_sams_; ++i) {
            if (dsol_[i] > 0.0) {
                XTdsol_ += dsol_[i] * X_.row(i);
            }
        }
        XTdsol_ *= inv_n_sams_;
    }

    for (int i = 0; i < n_feas_; ++i) {
        psol_[i] = val_sign(XTdsol_[i]) * inv_alpha_ *
            std::max(0.0, std::abs(XTdsol_[i]) - beta_);
    }
}

void ssvm_sifs::train(void) {
    int ind = 0;
    const double inv_nalpha_ = inv_n_sams_ * inv_alpha_;
    double delta_ind = 0.0;
    double p_theta_ind = 0.0;

    std::default_random_engine rg;
    std::uniform_int_distribution<> uni_dist(0, n_sams_ - 1);

    update_psol(true);
    compute_duality_gap(true, false);
 
    const auto ins_begin_it = std::begin(all_ins_index_);
    auto random_it = std::next(ins_begin_it, uni_dist(rg));
    for (iter_ = 1; iter_ < max_iter_ && duality_gap_ > tol_; ++iter_) {
        for (int jj = 0; jj < n_sams_; ++jj) {
            random_it = std::next(ins_begin_it, uni_dist(rg));
            ind = *random_it;

            p_theta_ind = dsol_[ind];
            delta_ind = (1 - gamma_ * p_theta_ind - (X_.row(ind) * psol_)(0)) /
                (gamma_ + Xi_norm_sq_[ind] * inv_nalpha_);
            delta_ind = std::max(-p_theta_ind, std::min(1.0 - p_theta_ind,
                                                        delta_ind));
            dsol_[ind] += delta_ind;
            XTdsol_ +=  (delta_ind * inv_n_sams_) * X_.row(ind);
            for (int kk = 0; kk < n_feas_; ++kk) {
                psol_[kk] = val_sign(XTdsol_[kk]) * inv_alpha_ *
                    std::max(0.0, std::abs(XTdsol_[kk]) - beta_);
            }
        }

        if (iter_ % chk_fre_ == 0) {
            compute_duality_gap(true, false);
        }
    }
}

void ssvm_sifs::train_sifs(const int& scr_option) {
    // we have closed form solution, if alpha > alpha_max(beta)
    if (alpha_ >= alpha_max_) {
        dsol_.setOnes();
        Eigen::ArrayXd temp =
            (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
            (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
        psol_ = temp * inv_alpha_;
        duality_gap_ = 0.0;
        return;
    }

    if (scr_option == 0) {  // alternative safe screening
                            // and performing sample screening first
        sifs(true);
    } else if (scr_option == 1) {  // alternative safe screening
                                   // and performing feature screening first
        sifs(false);
    } else if (scr_option == 2) {  // only sample screening
        iss();
    } else if (scr_option == 3) {  // only feature screening
        ifs();
    }

    XTdsol_.setZero();
    for (int i = 0; i < n_sams_; i++) {
        if (dsol_[i] > 0.0) {
            XTdsol_ += dsol_[i] * X_.row(i);
        }
    }
    XTdsol_ *= inv_n_sams_;
    for (auto &&kk : idx_Fc_vec_) {
        psol_[kk] = inv_alpha_ * val_sign(XTdsol_[kk]) *
            std::max(0.0, std::abs(XTdsol_[kk]) - beta_);
    }
    compute_duality_gap(true, false);

    int n_Dc = idx_Dc_vec_.size();
    if (n_Dc == 0) {
        std::cout << "  All samples screened. return" << std::endl;
        return;
    }

    int ind = 0;
    const double inv_nalpha_ = inv_n_sams_ * inv_alpha_;
    double delta_ind = 0.0;
    double p_theta_ind = 0.0;

    std::default_random_engine rg;
    std::uniform_int_distribution<> uni_dist(0, n_Dc - 1);

    const auto ins_begin_it = std::begin(idx_Dc_vec_);
    auto random_it = std::next(ins_begin_it, uni_dist(rg));
 
    for (iter_ = 1; iter_ < max_iter_ && duality_gap_ > tol_; ++iter_) {
        for (int jj = 0; jj < n_Dc; ++jj) {
            random_it = std::next(ins_begin_it, uni_dist(rg));
            ind = *random_it;

            p_theta_ind = dsol_[ind];
            delta_ind = (1 - gamma_ * p_theta_ind - (X_.row(ind) * psol_)(0)) /
                (gamma_ + Xi_norm_sq_[ind] * inv_nalpha_);
            delta_ind = std::max(-p_theta_ind, std::min(1.0 - p_theta_ind,
                                                        delta_ind));
            dsol_[ind] += delta_ind;
            XTdsol_ +=  (delta_ind * inv_n_sams_) * X_.row(ind);

            for (auto &&kk : idx_Fc_vec_) {
                psol_[kk] = inv_alpha_ * val_sign(XTdsol_[kk]) *
                    std::max(0.0, std::abs(XTdsol_[kk]) - beta_);
            }
        }

        if (iter_ % chk_fre_ == 0) {
            compute_duality_gap(true, false);
            //std::cout<< "    Iter: " << iter_ << " Primal obj: " << get_primal_obj()
            //         << " Dual obj: " << get_dual_obj()
            //         << " Duality gap: " << get_duality_gap() << std::endl;
        }
    }
}

void ssvm_sifs::clear_idx(void) {
    idx_Fc_vec_.clear();
    idx_Dc_vec_.clear();
    std::copy(std::begin(all_ins_index_), std::end(all_ins_index_),
              std::back_inserter(idx_Dc_vec_));
    std::copy(std::begin(all_fea_index_), std::end(all_fea_index_),
              std::back_inserter(idx_Fc_vec_));

    idx_Dc_.clear();
    idx_Fc_.clear();
    idx_Dc_.insert(std::begin(all_ins_index_), std::end(all_ins_index_));
    idx_Fc_.insert(std::begin(all_fea_index_), std::end(all_fea_index_));

    idx_nsv_L_.clear();
    idx_nsv_R_.clear();
    idx_F_.clear();

    approx_psol_r_F_sq_ = 0.0;
    approx_dsol_r_R_sq_ = 0.0;
    approx_dsol_r_L_sq_ = 0.0;
    approx_psol_r_sq_ = 0.0;
    approx_dsol_r_sq_ = 0.0;

    idx_Fc_flag_.setOnes();
    idx_Dc_flag_.setOnes();
}

void ssvm_sifs::set_alpha(const double& alpha, const bool& ws) {
    if (ws) {  // solved problem as warm start and reference solutions
        ref_alpha_ = alpha_;
        alpha_ = alpha;
        inv_alpha_ = 1.0 / alpha_;
        ref_psol_ = psol_;
        ref_dsol_ = dsol_;
    } else {  // no warm start, use naive solution as reference solutions
        alpha_ = alpha;
        inv_alpha_ = 1.0 / alpha_;
        ref_alpha_ = alpha_max_;
        Eigen::ArrayXd temp =
            (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
            (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
        ref_psol_ = temp * (1.0 / ref_alpha_);
        ref_dsol_.resize(n_sams_);
        ref_dsol_.setOnes();

        // set reference solutions as initial
        psol_ = ref_psol_;
        dsol_ = ref_dsol_;
    }

    dif_alpha_ratio_ = 0.5 * (alpha_ - ref_alpha_) / alpha_;
    sum_alpha_ratio_ = 0.5 * (alpha_ + ref_alpha_) / alpha_;
 
    duality_gap_ = std::numeric_limits<double>::max();
    iter_ = 0;
    fea_scr_time_ = 0.0;
    sam_scr_time_ = 0.0;
    scr_time_ = 0.0;

    clear_idx();
}

void ssvm_sifs::set_beta(const double& beta) {
    beta_ = beta;
    if (beta_ >= beta_max_) {
        std::cout << "beta >= beta_max: always naive solution" << std::endl;
    }
    Eigen::ArrayXd temp =
        (one_over_XTones_ > beta_).select(one_over_XTones_ - beta_, 0.0) +
        (one_over_XTones_ < -beta_).select(one_over_XTones_ + beta_, 0.0);
    alpha_max_ = (1.0/(1.0 - gamma_)) * (X_ * temp.matrix()).maxCoeff();
    if (alpha_max_ > 0) {
        ref_alpha_ = alpha_max_;
        ref_dsol_.resize(n_sams_);
        ref_dsol_.setOnes();
        ref_psol_ = temp * (1.0 / ref_alpha_);

        dif_alpha_ratio_ = 0.5 * (alpha_ - ref_alpha_) / alpha_;
        sum_alpha_ratio_ = 0.5 * (alpha_ + ref_alpha_) / alpha_;
    } else {
        ref_alpha_ = 0;
        ref_dsol_ = Eigen::VectorXd::Ones(n_sams_);
        ref_psol_ = Eigen::VectorXd::Zero(n_feas_);

        dif_alpha_ratio_ = 0.5;
        sum_alpha_ratio_ = 0.5;
    }
    
    duality_gap_ = std::numeric_limits<double>::max();
    iter_ = 0;
    fea_scr_time_ = 0.0;
    sam_scr_time_ = 0.0;
    scr_time_ = 0.0;

    clear_idx();
}

void ssvm_sifs::sample_screening(void) {
    auto start_time = sys_clk::now();

    const double L_coeff = 0.5 * ((2 * gamma_ - 1) * alpha_ + ref_alpha_) *
        inv_alpha_ * inv_gamma_;

    double psol_radius =
        std::sqrt(std::max(0.0, dif_alpha_ratio_ * dif_alpha_ratio_ *
                           approx_psol_r_sq_ - sum_alpha_ratio_ *
                           sum_alpha_ratio_ * approx_psol_r_F_sq_));

    double temp, xiw_comp_lb, xiw_comp_ub, Xi_Fc_sq, Xi_Fc_approx_psol_Fc;
    std::vector<int> new_nsv_L, new_nsv_R;

    for (auto &&i : idx_Dc_) {
        Xi_Fc_sq = 0.0;

        for (srm_iit it(X_, i); it; ++it) {
            if (idx_Fc_flag_[it.index()])
                Xi_Fc_sq += it.value() * it.value();
        }

        Xi_Fc_approx_psol_Fc = sum_alpha_ratio_ * (X_.row(i) * ref_psol_)(0);
        temp = psol_radius * std::sqrt(Xi_Fc_sq);
        xiw_comp_ub = 1 - Xi_Fc_approx_psol_Fc + temp;
        xiw_comp_lb = 1 - Xi_Fc_approx_psol_Fc - temp;

        if (xiw_comp_ub <  - 1e-9) {
            new_nsv_R.push_back(i);
        } else if (xiw_comp_lb > gamma_ + 1e-9) {
            new_nsv_L.push_back(i);
        }
    }

    for (auto &&i : new_nsv_R) {
        dsol_[i] = 0.0;
        idx_Dc_flag_[i] = 0.0;
        idx_nsv_R_.insert(i);
        idx_Dc_.erase(i);
        approx_dsol_c_[i] = 0.0;
        temp = dif_alpha_ratio_ * inv_gamma_ + sum_alpha_ratio_ * ref_dsol_[i];
        approx_dsol_r_R_sq_ += temp * temp;
    }
    for (auto &&i : new_nsv_L) {
        dsol_[i] = 1.0;
        idx_Dc_flag_[i] = 0.0;
        idx_nsv_L_.insert(i);
        idx_Dc_.erase(i);
        approx_dsol_c_[i] = 1.0;
        temp = L_coeff - sum_alpha_ratio_ * ref_dsol_[i];
        approx_dsol_r_L_sq_ += temp * temp;
    }

    auto end_time = sys_clk::now();
    sam_scr_time_ += static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void ssvm_sifs::feature_screening(void) {
    auto start_time = sys_clk::now();
    double dsol_radius =
        std::sqrt(std::max(0.0, dif_alpha_ratio_ * dif_alpha_ratio_ *
                           approx_dsol_r_sq_ - approx_dsol_r_L_sq_
                           - approx_dsol_r_R_sq_));

    double Xj_Dc_sq, temp, XjTdsol_ub;
    std::vector<int> new_iaf;
    double n_beta = n_sams_ * beta_;
    for (auto &&j : idx_Fc_) {
        Xj_Dc_sq = 0.0;

        for (scm_iit it(X_CM_, j); it; ++it) {
            if (idx_Dc_flag_[it.index()]) {
                Xj_Dc_sq += it.value() * it.value();
            }
        }

        temp = (approx_dsol_c_.transpose() * X_CM_.col(j))(0);
        XjTdsol_ub = std::abs(temp) + dsol_radius * std::sqrt(Xj_Dc_sq);
        if (XjTdsol_ub <= n_beta - 1e-9) {
            new_iaf.push_back(j);
        }
    }

    for (auto && j : new_iaf) {
        idx_Fc_flag_[j] = 0.0;
        idx_Fc_.erase(j);
        idx_F_.insert(j);
        psol_[j] = 0.0;
        approx_psol_r_F_sq_ += ref_psol_[j] * ref_psol_[j];
        ref_psol_[j] = 0.0;
    }

    auto end_time = sys_clk::now();
    fea_scr_time_ += static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void ssvm_sifs::sifs(const bool& sample_scr_first) {
    auto start_time = sys_clk::now();
    approx_dsol_c_ =
        (dif_alpha_ratio_ * inv_gamma_) + sum_alpha_ratio_ * ref_dsol_.array();
    approx_psol_r_sq_ = ref_psol_.squaredNorm();
    approx_dsol_r_sq_ = (ref_dsol_.array() - inv_gamma_).square().sum();
 
    int n_Dc = n_sams_;
    int n_Fc = n_feas_;
    int n_Dc_pre = n_Dc;
    int n_Fc_pre = n_Fc;
    int scr_iter = 0;
    
    while (true) {
        scr_iter++;
        n_Dc_pre = n_Dc;
        n_Fc_pre = n_Fc;
        if (sample_scr_first) {
            sample_screening();
            feature_screening();
        } else {
            feature_screening();
            sample_screening();
        }
        n_Dc = idx_Dc_.size();
        n_Fc = idx_Fc_.size();

        if ((n_Dc == n_Dc_pre) && (n_Fc == n_Fc_pre)) {
            break;
        }
        if (scr_max_iter_ > 0 && scr_iter >= scr_max_iter_) {
            break;
        }
    }

    idx_Dc_vec_.clear();
    idx_Fc_vec_.clear();

    for (auto &&j : idx_Dc_) {
        idx_Dc_vec_.push_back(j);
    }
    for (auto &&j : idx_Fc_) {
        idx_Fc_vec_.push_back(j);
    }
    auto end_time = sys_clk::now();
    scr_time_ = static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void ssvm_sifs::ifs(void) {
    auto start_time = sys_clk::now();
    approx_dsol_c_ =
        (dif_alpha_ratio_ * inv_gamma_) + sum_alpha_ratio_ * ref_dsol_.array();
    approx_psol_r_sq_ = ref_psol_.squaredNorm();
    approx_dsol_r_sq_ = (ref_dsol_.array() - inv_gamma_).square().sum();
 
    feature_screening();

    idx_Fc_vec_.clear();
    for (auto &&j : idx_Fc_) {
        idx_Fc_vec_.push_back(j);
    }
    auto end_time = sys_clk::now();
    scr_time_ = static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}

void ssvm_sifs::iss(void) {
    auto start_time = sys_clk::now();
    approx_dsol_c_ = (dif_alpha_ratio_ * inv_gamma_) + sum_alpha_ratio_ * ref_dsol_.array();
    approx_psol_r_sq_ = ref_psol_.squaredNorm();
    approx_dsol_r_sq_ = (ref_dsol_.array() - inv_gamma_).square().sum();
 
    sample_screening();

    idx_Dc_vec_.clear();
    for (auto &&j : idx_Dc_) {
        idx_Dc_vec_.push_back(j);
    }
    auto end_time = sys_clk::now();
    scr_time_ = static_cast<double>(
        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
}
