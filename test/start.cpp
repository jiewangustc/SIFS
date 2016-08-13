#include "ssvm_sifs.hpp"
#include <sdm/lib/utils.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>

/**
 * training elastic-net regularized smoothed sparse svm on a grid of
 * hyper-parameters, alpha and beta
 *
 *   min .5 * alpha * |w|^2 + beta * |w|_1 + (1/n) * sum_{i=1}^n l(1 - bar{x}_i^T w)
 *   where l(t) = 0, if t < 0
 *              = t^2 / (2 * gamma), if 0 <= t <= gamma
 *              = t - gamma/2, if t > gamma
 *         \bar{x}_i = y_i * x_i
 * 
 */

std::string input_fn;
double rbu, rbl, rau, ral;
int nbs, nas, max_iter, chk_fre, scr_max_iter;
double gam, tol;
double alpha, beta;
int task;

void print_usage(char *nm) {
    std::cout << "usage: " << nm << " train_file -task=task_type [option]..." << std::endl;
    std::cout
        << "  train_file : training file name (libsvm format). " << std::endl
        << "  -task=task_type : task type. " << std::endl
        << "    if =0, no screening" << std::endl
        << "    if =1, sifs, i.e., simultaneous inactive feature and sample screening" << std::endl
        << "    if =2, iss, i.e., inactive sample screening" << std::endl
        << "    if =3, ifs, i.e., inactive feature screening" << std::endl
        << "options:" << std::endl
        << "  -alpha=alpha (default 1e-3)" << std::endl
        << "  -beta=beta (default 1e-3)" << std::endl
        << "  -br.ub=rbu" << std::endl
        << "    maximum beta/beta_max (default=1)" << std::endl
        << "  -br.lb=rbl"  << std::endl
        << "    minimum beta/beta_max (default=0.05)" << std::endl
        << "  -b.ns=num_beta_split" << std::endl
        << "    the number of splits of beta (default=10)" << std::endl
        << "  -ar.ub=rau" << std::endl
        << "    maximum alpha/alpha_max(beta) (default=1)" << std::endl
        << "  -ar.lb=ral" << std::endl
        << "    mimimum alpha/alpha_max(beta) (default=0.01)" << std::endl
        << "  -a.ns=num_alpha_split" << std::endl
        << "    the number of splits of alpha (default=100)" << std::endl
        << "  -gam=gamma" << std::endl
        << "    smoothed hinge's parameter (default=0.05)"
        << "  -tol=tol" << std::endl
        << "    stopping tolerance for optimization (default=1e-9)" << std::endl
        << "  -max.iter=max_iter" << std::endl
        << "    max number of iteration for optimization (default=10000)" << std::endl
        << "  -chk.fre=chk_fre" << std::endl
        << "    frequence to check duality gap for optimization (defualt=1)" << std::endl
        << "  -scr.max.iter=scr_max_iter" << std::endl
        << "    maximum number of iterations for alternative safe screening" << std::endl
        << "    if =0, stop safe screening until no more inactive feature/sample found" << std::endl;

    std::cout << std::endl << std::endl;
    exit(0);
}

static char* parse(char *str, const char *arg) {
    if (strncmp(str, arg, strlen(arg)) == 0) {
        return (str + strlen(arg));
    }
    return 0;
}

void parse_command_line(int argc, char *argv[]) {
    // default parameters
    rbu = 1.0;
    rbl = 0.05;
    nbs = 10;
    rau = 1.0;
    ral = 0.01;
    nas = 100;
    max_iter = 10000;
    gam = 0.05;
    tol = 1e-9;
    chk_fre = 1;
    scr_max_iter = 0;
    alpha = 1e-3;
    beta = 1e-3;
    task = -1;

    char *s;
    if (argc < 3) {
        print_usage(argv[0]);
        return;
    } else {
        input_fn = argv[1];
        if ((s = parse(argv[2], "-task="))) {
            task = atoi(s);
        } else {
            print_usage(argv[0]);
            return;
        }
    }

    for (int i = 3; i < argc; ++i) {
        if ((s = parse(argv[i], "-alpha="))) {
            alpha = atof(s);
        } else if ((s = parse(argv[i], "-beta="))) {
            beta = atof(s);
        } else if ((s = parse(argv[i], "-br.ub="))) {
            rbu = atof(s);
        } else if ((s = parse(argv[i], "-br.lb="))) {
            rbl = atof(s);
        } else if ((s = parse(argv[i], "-b.ns="))) {
            nbs = atoi(s);
        } else if ((s = parse(argv[i], "-ar.ub="))) {
            rau = atof(s);
        } else if ((s = parse(argv[i], "-ar.lb="))) {
            ral = atof(s);
        } else if ((s = parse(argv[i], "-a.ns="))) {
            nas = atoi(s);
        } else if ((s = parse(argv[i], "-gam="))) {
            gam = atof(s);
        } else if ((s = parse(argv[i], "-tol="))) {
            tol = atof(s);
        } else if ((s = parse(argv[i], "-max.iter="))) {
            max_iter = atoi(s);
        } else if ((s = parse(argv[i], "-chk.fre="))) {
            chk_fre = atoi(s);
        } else if ((s = parse(argv[i], "-ss.max.iter="))) {
            scr_max_iter = atoi(s);
        } else {
            std::cout << " invalid option " << argv[i] << std::endl;
            print_usage(argv[0]);
            return;
        }
    }
 
    return;
}

int main(int argc, char *argv[])
{
    parse_command_line(argc, argv);

    if (argc < 3 || task < 0 || task > 3) {
        print_usage(argv[0]);
        exit(0);
    }

    ssvm_sifs solver(input_fn, alpha, beta, gam, tol, max_iter,
                     chk_fre, scr_max_iter);

    std::cout << "####################################################" << std::endl
              << "# of betas: " << nbs << ", beta/beta_max: " << rbl << " ~ " << rbu;
    std::cout << std::endl << std::endl;

    double beta_ub = solver.get_beta_max() * rbu;
    double beta_lb = solver.get_beta_max() * rbl;
    double beta_log_inter = (log10(beta_ub) - log10(beta_lb)) /
        static_cast<double>(nbs);
    double beta_now;
    
    for (int i = nbs - 1; i >= 0; --i) {
        beta_now = std::pow(10.0, log10(beta_lb) + 0.5 * beta_log_inter
                            + i * beta_log_inter);
        solver.set_beta(beta_now);

        std::cout << "===============================================" << std::endl
                  << "Solving "  << nbs - i << "-th beta, # of alphas: " << nas
                  << ", alpha/alpha_max: " << ral << " ~ " << rau << std::endl;

        double alpha_ub = solver.get_alpha_max() * rau;
        double alpha_lb = solver.get_alpha_max() * ral;
        double alpha_log_inter = (log10(alpha_ub) - log10(alpha_lb)) /
            static_cast<double>(nas);
        double alpha_now;

        for (int j = nas - 1; j >= 0; --j) {
            alpha_now = std::pow(10.0, log10(alpha_lb) + (j + 1) * alpha_log_inter);
            std::cout << "### solving " << nas -j << "-th alpha" << std::endl;

            if (j == nas - 1) {
                solver.set_alpha(alpha_now, false);
            } else {
                solver.set_alpha(alpha_now, true);
            }

            Eigen::VectorXd psol, dsol;
            Eigen::ArrayXd Xw_comp;
            int ias_R, ias_L, iaf;

            switch( task ) {
            case 0: {
                auto start_time = sys_clk::now();
                solver.train();
                auto end_time = sys_clk::now();

                double train_rt = 1e-3 * static_cast<double>(
                        std::chrono::duration_cast<mil_sec>(end_time - start_time).count());

                psol = solver.get_primal_sol();
                Xw_comp = 1.0 - (solver.X_ * psol).array();
                ias_R = (Xw_comp < 0 - 1e-8).select(
                    Eigen::VectorXd::Ones(solver.get_n_sams()), 0.0).sum();
                ias_L = (Xw_comp > gam + 1e-8).select(
                    Eigen::VectorXd::Ones(solver.get_n_sams()), 0.0).sum();
                iaf = (psol.array().abs() < 1e-8).select(
                    Eigen::VectorXd::Ones(solver.get_n_feas()), 0.0).sum();

                std::cout << "  iter: " << solver.get_iter()
                          << " duality_gap: " << solver.get_duality_gap()
                          << " solver_time (seconds): " << train_rt << std::endl;
 
                std::cout << "  GT: " << "ias_R: " << ias_R
                          << " ias_L: " << ias_L << " iaf: " << iaf << std::endl;
               break;
            }
            case 1: {
                auto start_time = sys_clk::now();
                solver.train_sifs(0);
                auto end_time = sys_clk::now();

                double train_rt = 1e-3 * static_cast<double>(
                    std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
                std::cout << "  iter: " << solver.get_iter()
                          << " duality_gap: " << solver.get_duality_gap()
                          << " scr_time (seconds): " << 1e-3 * solver.scr_time_
                          << " solver_time (seconds): "
                          << train_rt - 1e-3 * solver.scr_time_ << std::endl;
 
                std::cout << "  SIFS: " << "ias_R: " << solver.get_n_R()
                          << " ias_L:" << solver.get_n_L()
                          << " iaf: " << solver.get_n_F() << std::endl;
               break;
            }
            case 2: {
                auto start_time = sys_clk::now();
                solver.train_sifs(2);
                auto end_time = sys_clk::now();
                double train_rt = 1e-3 * static_cast<double>(
                    std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
                std::cout << "  iter: " << solver.get_iter()
                          << " duality_gap: " << solver.get_duality_gap()
                          << " scr_time (seconds): " << 1e-3 * solver.scr_time_
                          << " solver_time (seconds): "
                          << train_rt - 1e-3 * solver.scr_time_ << std::endl;
 
                std::cout << "  ISS: " << "ias_R: " << solver.get_n_R()
                          << " ias_L:" << solver.get_n_L() << std::endl;
                break;
            }
            case 3: {
                auto start_time = sys_clk::now();
                solver.train_sifs(3);
                auto end_time = sys_clk::now();

                double train_rt = 1e-3 * static_cast<double>(
                    std::chrono::duration_cast<mil_sec>(end_time - start_time).count());
                std::cout << "  iter: " << solver.get_iter()
                          << " duality_gap: " << solver.get_duality_gap()
                          << " scr_time (seconds): " << 1e-3 * solver.scr_time_
                          << " solver_time (seconds): "
                          << train_rt - 1e-3 * solver.scr_time_ << std::endl;
 
                std::cout << "  IFS: " << " iaf: " << solver.get_n_F() << std::endl;
               break;
            }
            }
        }
    }

    return 0;
}
