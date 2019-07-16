/* gdcpp.h
 *
 *     Author: Fabian Meyer
 * Created On: 12 Jul 2019
 *    License: MIT
 */

#ifndef GDCPP_GDCPP_H_
#define GDCPP_GDCPP_H_

#include <Eigen/Geometry>
#include <limits>
#include <iostream>
#include <iomanip>

namespace gdc
{
    typedef long int Index;

    /** Functor to compute forward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + eps) - f(x)) / eps
      *
      * The computation requires len(x) evaluations of the objective.
      */
    template<typename Scalar,
        typename Objective>
    struct ForwardDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar eps;
        Index threads;
        Objective *objective;

        ForwardDifferences()
            : ForwardDifferences(
                std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        ForwardDifferences(const Scalar eps)
            : eps(eps), threads(0), objective(nullptr)
        {

        }

        void operator()(const Vector &xval,
            const Scalar fval,
            Vector &gradient)
        {
            assert(objective != nullptr);

            Vector gradTmp;

            gradient.resize(xval.size());
            #pragma omp parallel for num_threads(threads)
            for(Index i = 0; i < xval.size(); ++i)
            {
                Vector xvalTmp = xval;
                xvalTmp(i) += eps;
                Scalar fvalNew = (*objective)(xvalTmp, gradTmp);

                gradient(i) = (fvalNew - fval) / eps;
            }
        }
    };

    /** Functor to compute backward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x) - f(x - eps)) / eps
      *
      * The computation requires len(x) evaluations of the objective.
      */
    template<typename Scalar,
        typename Objective>
    struct BackwardDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar eps;
        Index threads;
        Objective *objective;

        BackwardDifferences()
            : BackwardDifferences(
                std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        BackwardDifferences(const Scalar eps)
            : eps(eps), threads(0), objective(nullptr)
        {

        }

        void operator()(const Vector &xval,
            const Scalar fval,
            Vector &gradient)
        {
            assert(objective != nullptr);

            Vector gradTmp;

            gradient.resize(xval.size());
            #pragma omp parallel for num_threads(threads)
            for(Index i = 0; i < xval.size(); ++i)
            {
                Vector xvalTmp = xval;
                xvalTmp(i) -= eps;
                Scalar fvalNew = (*objective)(xvalTmp, gradTmp);

                gradient(i) = (fval - fvalNew) / eps;
            }
        }
    };

    /** Functor to compute central differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + 0.5 eps) - f(x - 0.5 eps)) / eps
      *
      * The computation requires 2 * len(x) evaluations of the objective.
      */
    template<typename Scalar,
        typename Objective>
    struct CentralDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar eps;
        Index threads;
        Objective *objective;

        CentralDifferences()
            : CentralDifferences(
                std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        CentralDifferences(const Scalar eps)
            : eps(eps),
            threads(1),
            objective(nullptr)
        {

        }

        void operator()(const Vector &xval,
            const Scalar,
            Vector &gradient)
        {
            assert(objective != nullptr);

            Vector gradTmp;

            Vector fvals(xval.size() * 2);
            #pragma omp parallel for num_threads(threads)
            for(Index i = 0; i < fvals.size(); ++i)
            {
                Index idx = i / 2;
                Vector xvalTmp = xval;
                if(i % 2 == 0)
                    xvalTmp(idx) += eps / 2;
                else
                    xvalTmp(idx) -= eps / 2;

                fvals(i) = (*objective)(xvalTmp, gradTmp);
            }

            gradient.resize(xval.size());
            for(Index i = 0; i < xval.size(); ++i)
                gradient(i) = (fvals(i * 2) - fvals(i * 2 + 1)) / eps;
        }
    };

    /** Dummy callback functor, which does nothing. */
    template<typename Scalar>
    struct NoCallback
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        bool operator()(const Index,
            const Vector &,
            const Scalar,
            const Vector &) const
        {
            return true;
        }
    };

    template<typename Scalar, typename Objective>
    struct ConstantStepSize
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar stepSize;
        Objective *objective;

        ConstantStepSize()
            : ConstantStepSize(0.7)
        {

        }

        ConstantStepSize(const Scalar stepSize)
            : stepSize(stepSize), objective(nullptr)
        {

        }

        Scalar operator()(const Vector &,
            const Scalar,
            const Vector &)
        {
            return stepSize;
        }
    };

    template<typename Scalar, typename Objective>
    struct LimitedChangeStep
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar changeLimit;
        Objective *objective;

        LimitedChangeStep()
            : LimitedChangeStep(0.1)
        {

        }

        LimitedChangeStep(const Scalar changeLimit)
            : changeLimit(changeLimit), objective(nullptr)
        {

        }

        Scalar operator()(const Vector &,
            const Scalar,
            const Vector &gradient)
        {
            return 1 / gradient.array().abs().maxCoeff();
        }
    };

    template<typename Scalar, typename Objective>
    struct BarzilaiBorweinStep
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Vector lastXval;
        Vector lastGradient;
        Objective *objective;

        BarzilaiBorweinStep()
            : lastXval(), lastGradient(), objective(nullptr)
        {

        }

        Scalar operator()(const Vector &xval,
            const Scalar,
            const Vector &gradient)
        {
            if(lastXval.size() == 0)
            {
                lastXval.setZero(xval.size());
                lastGradient.setZero(gradient.size());
            }

            Scalar num = ((xval - lastXval).transpose() * (gradient - lastGradient)) (0);
            num = std::abs(num);
            Scalar denom = (gradient - lastGradient).squaredNorm();

            lastGradient = gradient;
            lastXval = xval;
            if(denom == 0)
                return 1;
            else
                return num / denom;
        }
    };

    template<typename Scalar,
        typename Objective,
        typename FiniteDifferences=CentralDifferences<Scalar, Objective>>
    struct WolfeLineSearch
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Scalar decrease;
        Scalar c1;
        Scalar c2;
        Objective *objective;
        FiniteDifferences finiteDifferences;

        WolfeLineSearch()
            : WolfeLineSearch(0.9, 1e-4, 0.9)
        {

        }

        WolfeLineSearch(const Scalar decrease)
            : WolfeLineSearch(decrease, 1e-4, 0.9)
        {

        }

        WolfeLineSearch(const Scalar decrease, const Scalar c1, const Scalar c2)
            : decrease(decrease), c1(c1), c2(c2), objective(nullptr),
            finiteDifferences()
        {
            finiteDifferences.threads = 0;
        }

        Scalar evaluateObjective(const Vector &xval, Vector &gradient)
        {
            gradient.resize(0);
            Scalar fval = (*objective)(xval, gradient);
            if(gradient.size() == 0)
                finiteDifferences(xval, fval, gradient);
            return fval;
        }

        Scalar operator()(const Vector &xval,
            const Scalar fval,
            const Vector &gradient)
        {
            assert(objective != nullptr);
            finiteDifferences.objective = objective;
            finiteDifferences.threads = 0;

            Scalar stepSize = 1.0;

            Vector gradientTmp;
            Vector xvalTmp = xval + stepSize * -gradient;
            Scalar fvalTmp = evaluateObjective(xvalTmp, gradientTmp);
            Scalar gradientStep = (-gradient.transpose() * gradient)(0);

            while(fvalTmp > fval + c1 * stepSize * gradientStep
                || (-gradient.transpose() * gradientTmp)(0) > -c2 * gradientStep)
            {
                stepSize = decrease * stepSize;
                xvalTmp = xval + stepSize * -gradient;
                fvalTmp = evaluateObjective(xvalTmp, gradientTmp);
            }

            return stepSize;
        }
    };

    template<typename Scalar,
        typename Objective,
        typename StepSize=BarzilaiBorweinStep<Scalar, Objective>,
        typename Callback=NoCallback<Scalar>,
        typename FiniteDifferences=CentralDifferences<Scalar, Objective> >
    class GradientDescent
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        struct Result
        {
            Index iterations;
            bool converged;
            Scalar fval;
            Vector xval;
        };
    protected:
        Index maxIt_;
        Scalar minGradientLen_;
        Scalar minStepLen_;
        Scalar momentum_;
        Index verbosity_;
        Objective objective_;
        StepSize stepSize_;
        Callback callback_;
        FiniteDifferences finiteDifferences_;

        Scalar evaluateObjective(const Vector &xval, Vector &gradient)
        {
            gradient.resize(0);
            Scalar fval = objective_(xval, gradient);
            if(gradient.size() == 0)
                finiteDifferences_(xval, fval, gradient);
            return fval;
        }

        std::string vector2str(const Vector &vec) const
        {
            std::stringstream ss1;
            ss1 << std::fixed << std::showpoint << std::setprecision(6);
            std::stringstream ss2;
            ss2 << '[';
            for(Index i = 0; i < vec.size(); ++i)
            {
                ss1 << vec(i);
                ss2 << std::setfill(' ') << std::setw(10) << ss1.str();
                if(i != vec.size() - 1)
                    ss2 << ' ';
                ss1.str("");
            }
            ss2 << ']';

            return ss2.str();
        }

    public:

        GradientDescent()
            : maxIt_(0), minGradientLen_(static_cast<Scalar>(1e-9)),
            minStepLen_(static_cast<Scalar>(1e-9)), momentum_(0),
            verbosity_(0), objective_(), stepSize_(), callback_(),
            finiteDifferences_()
        {

        }

        ~GradientDescent()
        {

        }

        void setThreads(const Index threads)
        {
            finiteDifferences_.threads = threads;
        }

        void setMaxIterations(const Index iterations)
        {
            maxIt_ = iterations;
        }

        void setObjective(const Objective &objective)
        {
            objective_ = objective;
        }

        void setCallback(const Callback &callback)
        {
            callback_ = callback;
        }

        void setNumericalEpsilon(const Scalar eps)
        {
            finiteDifferences_.eps = eps;
        }

        void setMinGradientLength(const Scalar gradientLen)
        {
            minGradientLen_ = gradientLen;
        }

        void setMinStepLength(const Scalar stepLen)
        {
            minStepLen_ = stepLen;
        }

        void setStepSize(const StepSize stepSize)
        {
            stepSize_ = stepSize;
        }

        void setMomentum(const Scalar momentum)
        {
            momentum_ = momentum;
        }

        void setVerbosity(const Index verbosity)
        {
            verbosity_ = verbosity;
        }

        Result minimize(const Vector &initialGuess)
        {
            finiteDifferences_.objective = &objective_;
            stepSize_.objective = &objective_;

            Vector gradient;
            Vector xval = initialGuess;

            Scalar fval = evaluateObjective(xval, gradient);
            Scalar gradientLen = gradient.norm();
            Scalar stepSize = stepSize_(xval, fval, gradient);
            Vector step = (1 - momentum_) * stepSize * gradient;
            Scalar stepLen = step.norm();

            Index iterations = 0;
            while((maxIt_ <= 0 || iterations < maxIt_) &&
                gradientLen >= minGradientLen_ &&
                stepLen >= minStepLen_)
            {
                xval -= step;
                fval = evaluateObjective(xval, gradient);
                gradientLen = gradient.norm();
                // update step according to step size and momentum
                stepSize = stepSize_(xval, fval, gradient);
                step = momentum_ * step + (1 - momentum_) * stepSize * gradient;
                stepLen = step.norm();

                if(verbosity_ > 0)
                {
                    std::stringstream ss;
                    ss << "it=" << std::setfill('0')
                        << std::setw(4) << iterations
                        << std::fixed << std::showpoint << std::setprecision(6)
                        << "    gradlen=" << gradientLen
                        << "    stepsize=" << stepSize
                        << "    steplen=" << stepLen
                        << "    fval=" << fval;
                    if(verbosity_ > 1)
                        ss << "    xval=" << vector2str(xval);
                    if(verbosity_ > 2)
                        ss << "    gradient=" << vector2str(gradient);
                    if(verbosity_ > 3)
                        ss << "    step=" << vector2str(step);
                    std::cout << ss.str() << std::endl;;
                }

                if(!callback_(iterations, xval, fval, gradient))
                    break;

                ++iterations;
            }

            Result result;
            result.xval = xval;
            result.fval = fval;
            result.iterations = iterations;
            result.converged = gradientLen < minGradientLen_ ||
                stepLen < minStepLen_;

            return result;
        }
    };
}

#endif
