#pragma once

#include <limits>
#include <cmath>
#include <cassert>
#include <tuple>

namespace LPMP {

    // TODO: add operator +=, remove update

    template<typename REAL>
    struct exp_sum {
        exp_sum() {}
        exp_sum(const REAL _sum, const REAL _max) : sum(_sum), max(_max) {}
        exp_sum(const REAL x);
        REAL sum = 0.0;
        REAL max = -std::numeric_limits<REAL>::infinity();

        void update(const exp_sum o);
        exp_sum<REAL> operator+(const exp_sum<REAL>& o) const;
        exp_sum<REAL>& operator+=(const exp_sum<REAL>& o);
        exp_sum<REAL> operator*(const exp_sum<REAL>& o) const;
        exp_sum<REAL>& operator*=(const exp_sum<REAL>& o);
        exp_sum<REAL> operator*(const REAL x) const;
        exp_sum<REAL>& operator*=(const REAL x);
        bool operator==(const exp_sum<REAL>& o) const;
        REAL log() const;
        operator std::tuple<REAL,REAL>() const { return {sum, max}; }
    };

    template<typename REAL>
        double exp_sum_diff_log(const exp_sum<REAL> a, const exp_sum<REAL> b)
        {
            return std::log(a.sum/b.sum) + a.max-b.max;
        }

    template<typename REAL>
        exp_sum<REAL> operator*(const REAL x, const exp_sum<REAL> es)
        {
            return es * x;
        }

    // encapsulation in exp_sum
    template<typename REAL>
        exp_sum<REAL>::exp_sum(const REAL x)
        {
            sum = 1.0;
            max = x;
        }

    template<typename REAL>
    void exp_sum<REAL>::update(const exp_sum o)
    {
        assert(std::isfinite(sum));
        assert(sum >= 0.0);
        assert(!std::isnan(max));
        
        if(o.sum == 0.0)
        {
            //assert(o.max == -std::numeric_limits<REAL>::infinity());
            return;
        }
        
        if(max > o.max)
            sum += o.sum * std::exp(o.max - max);
        else
        {
            sum *= std::exp(max - o.max);
            sum += o.sum;
            max = o.max;
        }

        assert(std::isfinite(sum));
        assert(!std::isnan(max));
    }

    template<typename REAL>
        exp_sum<REAL> exp_sum<REAL>::operator+(const exp_sum<REAL>& o) const
        {
            exp_sum<REAL> es = *this;
            es.update(o);
            return es;
        }

    template<typename REAL>
        exp_sum<REAL>& exp_sum<REAL>::operator+=(const exp_sum<REAL>& o)
        {
            *this = *this + o;
            return *this;
        }

    template<typename REAL>
        exp_sum<REAL> exp_sum<REAL>::operator*(const exp_sum<REAL>& o) const
        {
            exp_sum<REAL> es = *this;
            es.sum *= o.sum;
            es.max += o.max;
            return es;
        }

    template<typename REAL>
        exp_sum<REAL>& exp_sum<REAL>::operator*=(const exp_sum<REAL>& o)
        {
            *this = *this * o;
            return *this;
        }

    template<typename REAL>
        exp_sum<REAL> exp_sum<REAL>::operator*(const REAL x) const
        {
            assert(x > 0.0);
            exp_sum<REAL> es = *this;
            es.sum *= x;
            return es;
        }

    template<typename REAL>
        exp_sum<REAL>& exp_sum<REAL>::operator*=(const REAL x)
        {
            assert(x > 0.0);
            sum *= x;
            return *this;
        }

    template<typename REAL>
        bool exp_sum<REAL>::operator==(const exp_sum<REAL>& o) const
        {
            return sum == o.sum && max == o.max;
        }

    template<typename REAL>
        REAL exp_sum<REAL>::log() const
        {
            return std::log(sum) + max;
        }

}
