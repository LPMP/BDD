#include "transitive_closure_dag.h"

#include <queue>
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    two_dim_variable_array<size_t> convert_arc_list(const std::vector<std::array<size_t, 2>> &arc_list)
    {
        const size_t nr_nodes = [&]()
        {
            size_t nr_nodes = 0;
            for (const auto [i, j] : arc_list)
                nr_nodes = std::max({nr_nodes, i + 1, j + 1});
            return nr_nodes;
        }();
        std::vector<size_t> out_arc_counter(nr_nodes, 0);

        for (const auto [i, j] : arc_list)
            out_arc_counter[i]++;

        two_dim_variable_array<size_t> arcs(out_arc_counter);
        assert(arcs.size() == out_arc_counter.size());
        std::fill(out_arc_counter.begin(), out_arc_counter.end(), 0);

        for(const auto [i,j] : arc_list)
            arcs(i, out_arc_counter[i]++) = j;

        return arcs;
    }

    // Kahn's algorithm
    std::vector<size_t> topological_sorting(const two_dim_variable_array<size_t> &dag)
    {
        std::vector<size_t> incoming_counter;
        incoming_counter.resize(dag.size());
        for(size_t i=0; i<dag.size(); ++i)
            for (size_t c = 0; c < dag.size(i); ++c)
                incoming_counter[dag(i,c)] += 1;

        std::queue<size_t> q;
        for (size_t i = 0; i < incoming_counter.size(); ++i)
            if (incoming_counter[i] == 0)
                q.push(i);
        assert(q.size() > 0);

        std::vector<size_t> vars_ordered;
        vars_ordered.reserve(dag.size());

        while(!q.empty())
        {
            const size_t i = q.front();
            q.pop();
            vars_ordered.push_back(i);
            for(size_t c=0; c<dag.size(i); ++c)
            {
                const size_t j = dag(i,c);
                assert(incoming_counter[j] > 0);
                incoming_counter[j]--;
                if (incoming_counter[j] == 0)
                    q.push(j);
            }
        }

        for(size_t i=0; i<incoming_counter.size(); ++i)
            assert(incoming_counter[i] == 0);

        assert(vars_ordered.size() == dag.size());
        return vars_ordered;
    }

    bool interval_rep::is_reduced() const
    {
        for (const auto [i, j] : interval_)
            assert(i <= j);

        for(size_t i=1; i<interval_.size(); ++i)
            if (!(interval_[i - 1][1] + 1 < interval_[i][0]))
                return false;
        
        return true;
    }

    bool interval_rep::operator[](const size_t i) const
    {
        std::ptrdiff_t lo = 0;
        std::ptrdiff_t hi = interval_.size() - 1;

        while(lo <= hi)
        {
            const std::ptrdiff_t middle = (lo + hi) / 2;
            if (i < interval_[middle][0])
                hi = middle - 1;
            else if (i > interval_[middle][1])
                lo = middle + 1;
            else
            {
                assert(i >= interval_[middle][0] && i <= interval_[middle][1]);
                return true;
            }
        }

        return false;
    }

    interval_rep merge(const interval_rep &i1, const interval_rep &i2)
    {
        assert(i1.is_reduced());
        assert(i2.is_reduced());

        if(i1.interval_.size() == 0)
            return i2;
        if(i2.interval_.size() == 0)
            return i1;

        interval_rep u;

        auto append = [&](const std::array<size_t,2>& intr) {
            assert(u.interval_.size() > 0);
            if (intr[0] <= u.interval_.back()[1] + 1)
                u.interval_.back()[1] = std::max(u.interval_.back()[1], intr[1]);
            else
                u.interval_.push_back(intr);
        };

        size_t c1 = 0;
        size_t c2 = 0;

        // push first interval
        if (i1.interval_[0][0] < i2.interval_[0][0])
        {
            u.interval_.push_back(i1.interval_[0]);
            c1++;
        }
        else
        {
            u.interval_.push_back(i2.interval_[0]);
            c2++;
        }

        while(c1 < i1.interval_.size() && c2 < i2.interval_.size())
        {
            if (i1.interval_[c1][0] < i2.interval_[c2][0])
            {
                append(i1.interval_[c1]);
                c1++;
            }
            else
            {
                append(i2.interval_[c2]);
                c2++;
            }
        }

        assert(c1 == i1.interval_.size() || c2 == i2.interval_.size());

        for(; c1<i1.interval_.size(); ++c1)
            append(i1.interval_[c1]);
        for(; c2<i2.interval_.size(); ++c2)
            append(i2.interval_[c2]);

        assert(u.is_reduced());

        return u;

        /*
        auto interval_contained = [&]() -> bool {
            if(i1.interval_[c1][0] <= i2.interval_[c2][0] && i1.interval_[c1][1] >= i2.interval_[c2][1])
            {
                u.interval_.push_back(i1.interval_[c1]);
                c1++;
                c2++;
                return true;
            }
            if(i2.interval_[c2][0] <= i1.interval_[c1][0] && i2.interval_[c2][1] >= i1.interval_[c1][1])
            {
                u.interval_.push_back(i1.interval_[c1]);
                c1++;
                c2++;
                return true;
            }
            return false;
        };

        auto extend_interval = [&]( {
            if()

        };

        auto copy_interval = [&]() {

        };

        auto i1_ahead = [&](size_t h, auto &i2_ahead_ref) -> void
        {
            auto i1_ahead_impl = [&](const size_t h, auto &i1_ahead_ref, auto &i2_ahead_ref)
            {
                while (c2 < i2.interval_.size() && i2.interval_[c2][1] <= h)
                    c2++;
                if (c2 == i2.interval_.size())
                    u.interval_.back()[1] = h;
                else if (i2.interval_[c2][0] - 1 <= h)
                    i2_ahead_ref(i2.interval_[c2++][1], i1_ahead_ref);
                else
                    u.interval_.back()[1] = h;
            };
            return i1_ahead_impl(h, i1_ahead_impl, i2_ahead_ref);
        };

        auto i2_ahead = [&](const size_t h, auto &i1_ahead_ref) -> void
        {
            auto i2_ahead_impl = [&](const size_t h, auto &i1_ahead_ref, auto &i2_ahead_ref)
            {
                while (c1 < i1.interval_.size() && i1.interval_[c1][1] <= h)
                    c1++;
                if (c1 == i1.interval_.size())
                    u.interval_.back()[1] = h;
                else if (i1.interval_[c1][0] - 1 <= h)
                    i1_ahead_ref(i1.interval_[c1++][1], i2_ahead_ref);
                else
                    u.interval_.back()[1] = h;
            };
            return i2_ahead_impl(h, i1_ahead_ref, i2_ahead_impl);
        };

        while (c1 < i1.interval_.size() && c2 < i2.interval_.size())
        {
            if(i1.interval_[c1][1] < i2.interval_[c2][0] - 1)
            {
                u.interval_.push_back(i1.interval_[c1]);
                c1++;
            }
            else if(i2.interval_[c2][1] < i1.interval_[c1][0] - 1)
            {
                u.interval_.push_back(i2.interval_[c2]);
                c2++;
            }
            else
            {
                if (i1.interval_[c1][0] < i2.interval_[c2][0])
                    u.interval_.push_back(i1.interval_[c1]);
                else
                    u.interval_.push_back(i2.interval_[c2]);
                if(i1.interval_[c1][1] > i2.interval_[c2][1])
                {
                    const size_t h = i1.interval_[c1][1];
                    c1++;
                    c2++;
                                    }
                else if(i1.interval_[c1][1] < i2.interval_[c2][1])
                {
                    const size_t h = i2.interval_[c2][1];
                    c1++;
                    c2++;
                }
                else
                {
                    u.interval_.back()[1] = i2.interval_[c2][1];
                    c1++;
                    c2++;
                }
            }
        }

        // copy over remaining part
        assert(c1 == i1.interval_.size() || c2 == i2.interval_.size());
        for(; c1<i1.interval_.size(); ++c1)
        {

        }

        assert(is_reduced(u));

        return u;
        */
    }

    std::ostream& operator<< (std::ostream &s, const interval_rep &intr)
    {
        if (intr.interval_.size() == 0)
            return s << "{}";

        s << "{[" << (intr.interval_[0][0]) << "," << (intr.interval_[0][1]) << "]";
        for(size_t i=1; i<intr.interval_.size(); ++i)
            s << ", [" << (intr.interval_[i][0]) << "," << (intr.interval_[i][1]) << "]";
        return s << "}";
    }

    transitive_closure::transitive_closure(const std::vector<std::array<size_t,2>>& arcs)
    {
        assert(arcs.size() > 0);
        const auto dag = convert_arc_list(arcs);

        const auto top_order = topological_sorting(dag);
        inv_top_order_.resize(dag.size());
        for (size_t i = 0; i < top_order.size(); ++i)
            inv_top_order_[top_order[i]] = i;

        intervals_.reserve(dag.size());
        std::vector<size_t> suc;
        for(size_t i=0; i<dag.size(); ++i)
        {
            suc.push_back(inv_top_order_[i]); // self-loop
            for(size_t c=0; c<dag.size(i); ++c)
                suc.push_back(inv_top_order_[dag(i,c)]);
            intervals_.push_back(interval_rep(suc.begin(), suc.end()));
            suc.clear();
        }

        for (auto it = top_order.rbegin(); it != top_order.rend(); ++it)
        {
            const size_t i = *it;
            // TODO: more efficient merging?
            for(size_t c=0; c<dag.size(i); ++c)
            {
                const size_t j = dag(i, c);
                intervals_[i] = merge(intervals_[i], intervals_[j]);
            }
        }
    }

    size_t transitive_closure::nr_nodes() const
    {
        return intervals_.size();
    }

    bool transitive_closure::operator()(const size_t i, const size_t j) const
    {
        assert(i < intervals_.size());
        assert(j < intervals_.size());
        return intervals_[i][inv_top_order_[j]];
    }
}