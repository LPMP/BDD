#include "exp_sum.h"
#include "test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    // distributivity
    test(exp_sum<float>(1.0) + exp_sum<float>(2.0) == exp_sum<float>(2.0) + exp_sum<float>(1.0));

    // approximates max
    exp_sum<float> es = exp_sum<float>(1.0) + exp_sum<float>(2.0);
    test(std::abs(es.log() - std::log(std::exp(1.0) + std::exp(2.0))) <= 1e-6);
    test(2.0 <= es.log());
    test(es.log() <= 2.0 + std::log(2));

    // correctly sets max
    const auto [s, max] = exp_sum<float>(1.0) + exp_sum<float>(2.0);
    test(max == 2.0);

    // correct multiplication
    {
        const exp_sum<double> m1 = exp_sum<double>(1.0);
        const exp_sum<double> m2 = exp_sum<double>(2.0);
        const exp_sum<double> m12 = m1*m2;
        const exp_sum<double> m3 = exp_sum<double>(3.0);
        test(m12 == m3);
    }

    // multiplication + log correct
    {
        const exp_sum<double> m(3.0);
        test((m*m).log() == exp_sum<double>(6.0).log());
    }

    // multiplication by constant
    {
        const exp_sum<double> m3(3.0);
        const exp_sum<double> m3p3 = m3 + m3;
        const exp_sum<double> m3x2 = 2.0*m3;
        test(2.0*m3 == m3*2.0);
        test(m3x2.log() == m3p3.log());
    }
}
