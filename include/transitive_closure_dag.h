#pragma once

#include <vector>
#include <array>
#include <ostream>

namespace LPMP {

    class interval_rep
    {
    public:
        interval_rep() {}
        interval_rep(const size_t i) { interval_ = {std::array<size_t,2>{i,i}}; }
        template <typename ITER>
        interval_rep(ITER begin, ITER end);
        bool operator[](const size_t i) const;
        bool is_reduced() const;
        friend interval_rep merge(const interval_rep &i1, const interval_rep &i2);
        friend std::ostream &operator<<(std::ostream &out, const interval_rep &intr);
    private:
        std::vector<std::array<size_t,2>> interval_; // x is in interval if there exists an interval x \in [i,j]
    };

    interval_rep merge(const interval_rep &i1, const interval_rep &i2);

    std::ostream& operator<< (std::ostream &out, const interval_rep &intr);

    class transitive_closure
    {
    public:
        transitive_closure(const std::vector<std::array<size_t, 2>> &dag);
        size_t nr_nodes() const;
        bool operator()(const size_t i, const size_t j) const;

    private:
        std::vector<size_t> inv_top_order_;
        std::vector<interval_rep> intervals_;
    };

    template <typename ITER>
    interval_rep::interval_rep(ITER begin, ITER end)
    {
        std::vector<size_t> elems(begin, end);
        std::sort(elems.begin(), elems.end());
        assert(std::unique(elems.begin(), elems.end()) == elems.end());
        if(elems.size() == 0)
            return;

        interval_.push_back({elems[0], elems[0]});
        for(size_t i=1; i<elems.size(); ++i)
            if(interval_.back()[1] + 1 == elems[i])
                interval_.back()[1]++;
            else
                interval_.push_back({elems[i], elems[i]});

        assert(is_reduced());
    }
}