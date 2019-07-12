/*
 * finite_differences.h
 *
 * Author: Fabian Meyer
 * Created On: 12 Jul 2019
 */

#ifndef GDC_FINITE_DIFFERENCES_H_
#define GDC_FINITE_DIFFERENCES_H_

#include <Eigen/Geometry>
#include <limits>

namespace gdc
{
    typedef long int Index;

    template<typename Scalar,
        typename Objective>
    struct ForwardDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Objective objective;
        Scalar eps;

        ForwardDifferences()
            : objective(),
            eps(std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        ForwardDifferences(const Scalar eps)
            : objective(), eps(eps)
        {

        }

        void operator()(const Vector &state,
            const Scalar fval,
            Vector &gradient) const
        {
            Vector tmp;
            Vector nstate = state;

            gradient.resize(state.size());
            for(Index i = 0; i < state.size(); ++i)
            {
                nstate(i) += eps;
                Scalar fvalNew = objective(nstate, tmp);
                nstate(i) = state(i);

                gradient(i) = (fvalNew - fval) / eps;
            }
        }
    };

    template<typename Scalar,
        typename Objective>
    struct BackwardDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Objective objective;
        Scalar eps;

        BackwardDifferences()
            : objective(),
            eps(std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        BackwardDifferences(const Scalar eps)
            : objective(), eps(eps)
        {

        }

        void operator()(const Vector &state,
            const Scalar fval,
            Vector &gradient) const
        {
            Vector nstate = state;
            Vector tmp;

            gradient.resize(state.size());
            for(Index i = 0; i < state.size(); ++i)
            {
                nstate(i) -= eps;
                Scalar fvalNew = objective(nstate, tmp);
                nstate(i) = state(i);

                gradient(i) = (fval - fvalNew) / eps;
            }
        }
    };

    template<typename Scalar,
        typename Objective>
    struct CentralDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Objective objective;
        Scalar eps;

        CentralDifferences()
            : objective(), eps(std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        CentralDifferences(const Scalar eps)
            : objective(), eps(eps)
        {

        }

        void operator()(const Vector &state,
            const Scalar,
            Vector &gradient) const
        {
            Vector nstate = state;
            Vector tmp;

            gradient.resize(state.size());
            for(Index i = 0; i < state.size(); ++i)
            {
                nstate(i) = state(i) + eps / 2;
                Scalar fvalA = objective(nstate, tmp);
                nstate(i) = state(i) - eps / 2;
                Scalar fvalB = objective(nstate, tmp);
                nstate(i) = state(i);

                gradient(i) = (fvalA - fvalB) / eps;
            }
        }
    };
}

#endif
