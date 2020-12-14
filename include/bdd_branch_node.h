#pragma once

#include <array>
#include <numeric>
#include <cstddef>
#include <type_traits>
#include "bdd_variable.h"

namespace LPMP {

    ///////////////////////////////
    // Base Template Branch Node //
    ///////////////////////////////

    template<typename DERIVED>
    class bdd_branch_node {
        public:
            DERIVED* low_outgoing = nullptr;
            DERIVED* high_outgoing = nullptr;
            //DERIVED* first_low_incoming = nullptr;
            //DERIVED* first_high_incoming = nullptr;
            //DERIVED* next_low_incoming = nullptr;
            //DERIVED* next_high_incoming = nullptr;

            // From C++20
            friend bool operator==(const bdd_branch_node<DERIVED>& x, const bdd_branch_node<DERIVED>& y);

            static bool is_terminal(DERIVED* p) { return p == terminal_0() || p == terminal_1(); }
            //bool is_first() const { return first_low_incoming == nullptr && first_high_incoming == nullptr; }
            bool is_dead_end() const { return low_outgoing == terminal_0() && high_outgoing == terminal_0(); }
            // bool is_initial_state() const { return *this == DERIVED{}; }
            
            constexpr static DERIVED* terminal_0();
            static inline DERIVED tmp_terminal_0 = {terminal_0(), terminal_0()};//, terminal_0(), terminal_0(), terminal_0(), terminal_0()};

            constexpr static DERIVED* terminal_1();
            static inline DERIVED tmp_terminal_1 = {terminal_1(), terminal_1()}; //, terminal_1(), terminal_1(), terminal_1(), terminal_1()};
    };

    template<typename DERIVED>
        class bdd_branch_node_incoming_pointers : public bdd_branch_node<DERIVED> {
            public:
                DERIVED* first_low_incoming = nullptr;
                DERIVED* first_high_incoming = nullptr;
                DERIVED* next_low_incoming = nullptr;
                DERIVED* next_high_incoming = nullptr;

                bool is_first() const { return first_low_incoming == nullptr && first_high_incoming == nullptr; }

                friend bool operator==(const bdd_branch_node_incoming_pointers<DERIVED>& x, const bdd_branch_node_incoming_pointers<DERIVED>& y);
    };

    template<typename BDD_NODE_TYPE>
        struct has_incoming_pointers
        {
            constexpr static bool value = std::is_convertible<BDD_NODE_TYPE*, bdd_branch_node_incoming_pointers<BDD_NODE_TYPE>*>::value;
        };

    template<typename DERIVED>
        constexpr DERIVED* bdd_branch_node<DERIVED>::terminal_0()
        {
            return &tmp_terminal_0; 
        }

    template<typename DERIVED>
        constexpr DERIVED* bdd_branch_node<DERIVED>::terminal_1()
        {
            return &tmp_terminal_1; 
        }

    template<typename DERIVED>
    bool operator==(const bdd_branch_node<DERIVED>& x, const bdd_branch_node<DERIVED>& y)
    {
        const bool equal = (x.low_outgoing == y.low_outgoing &&
            x.high_outgoing == y.high_outgoing &&
            x.first_low_incoming == y.first_low_incoming &&
            x.first_high_incoming == y.first_high_incoming &&
            x.next_low_incoming == y.next_low_incoming &&
            x.next_high_incoming == y.next_high_incoming);
        return equal;
    }

    template<typename DERIVED>
    bool operator==(const bdd_branch_node_incoming_pointers<DERIVED>& x, const bdd_branch_node_incoming_pointers<DERIVED>& y)
    {
        if(!static_cast<bdd_branch_node<DERIVED>*>(x) == static_cast<bdd_branch_node<DERIVED>*>(y))
            return false;
        const bool equal = (x.first_low_incoming == y.first_low_incoming &&
            x.first_high_incoming == y.first_high_incoming &&
            x.next_low_incoming == y.next_low_incoming &&
            x.next_high_incoming == y.next_high_incoming);
        return equal;
    }

    template<typename DERIVED>
    void check_bdd_branch_node(const bdd_branch_node<DERIVED>& bdd, const bool last_variable = false, const bool first_variable = false)
    {
        return;
#ifdef NDEBUG
        return;
#endif
        // need not hold true if variable is always infeasible
        //assert(!bdd.is_first() || (bdd.low_outgoing != bdd_branch_node<DERIVED>::terminal_0() && bdd.high_outgoing != bdd_branch_node<DERIVED>::terminal_0()));
        assert(bdd.low_outgoing != nullptr);
        assert(bdd.low_outgoing == DERIVED::terminal_0() || bdd.low_outgoing == DERIVED::terminal_1() || bdd.low_outgoing > &bdd);
        assert(bdd.high_outgoing != nullptr);
        assert(bdd.high_outgoing == DERIVED::terminal_0() || bdd.high_outgoing == DERIVED::terminal_1() || bdd.high_outgoing > &bdd);

        if(last_variable) {
            assert(bdd.low_outgoing == DERIVED::terminal_0() || bdd.low_outgoing == DERIVED::terminal_1());
            assert(bdd.high_outgoing == DERIVED::terminal_0() || bdd.high_outgoing == DERIVED::terminal_1());
        }
    }

    template<typename DERIVED>
    void check_bdd_branch_node(const bdd_branch_node_incoming_pointers<DERIVED>& bdd, const bool last_variable = false, const bool first_variable = false)
    {
        // TODO: activate
        assert(false);
        //check_bdd_branch_node(*static_cast<bdd_branch_node<DERIVED>*>(&bdd), last_variable, first_variable);

        /*
        if(first_variable) {
            assert(bdd.first_low_incoming == nullptr);
            assert(bdd.first_high_incoming == nullptr);
        } 
        if(bdd.first_low_incoming != nullptr) {
            bdd_branch_node<DERIVED>* cur = bdd.first_low_incoming;
            while(cur != nullptr) {
                assert(cur < &bdd);
                assert(cur->low_outgoing == &bdd);
                cur = cur->next_low_incoming;
            }
        }
        if(bdd.first_high_incoming != nullptr) {
            bdd_branch_node<DERIVED>* cur = bdd.first_high_incoming;
            while(cur != nullptr) {
                assert(cur < &bdd);
                assert(cur->high_outgoing == &bdd);
                cur = cur->next_high_incoming;
            }
        }
        */
    }


    //////////////////////////////
    // Optimization Branch Node //
    //////////////////////////////

    // BDD_BASE can be either bdd_branch_node or bdd_branch_node_incoming_pointers
    template<typename DERIVED>
    class bdd_branch_node_opt_base : public bdd_branch_node<DERIVED> {
        public:
            double* variable_cost = nullptr;
            double m = 0.0; // intermediate value of shortest path from either terminal or first node (depending on algorithm state)

            // From C++20
            friend bool operator==(const bdd_branch_node_opt_base<DERIVED>& x, const bdd_branch_node_opt_base<DERIVED>& y);

            void backward_step();
            void forward_step();

            // Debug functions for checking correctness of forward and backward step
            double cost_from_first() const;
            double cost_from_terminal() const;

            std::array<double,2> min_marginal() const;
            std::array<double,2> min_marginal_debug() const;
    };

    //template<typename DERIVED>
    //    DERIVED bdd_branch_node<bdd_branch_node_opt_base<DERIVED>>::tmp_terminal_0 = {};// {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.0};

    template<typename DERIVED>
    bool operator==(const bdd_branch_node_opt_base<DERIVED>& x, const bdd_branch_node_opt_base<DERIVED>& y)
    {
        const bool equal = (x.low_outgoing == y.low_outgoing &&
                x.high_outgoing == y.high_outgoing &&
                x.variable_cost == y.variable_cost &&
                x.m == y.m);
        return equal;
    }

    /*
    template<typename DERIVED>
    void bdd_branch_node_opt_base<DERIVED>::forward_step()
    {
        check_bdd_branch_node(*this);

        if(this->is_first()) {
            m = 0.0;
            //std::cout << "forward step m for " << this << " = " << m << "\n";
            return;
        }

        m = std::numeric_limits<double>::infinity();

        // iterate over all incoming low edges 
        {
            auto* cur = this->first_low_incoming;
            while(cur != nullptr) {
                //m = std::min(m, cur->m + *cumulative_sum - *(cur->cumulative_sum));
                m = std::min(m, cur->m);
                cur = cur->next_low_incoming;
            }
        }

        // iterate over all incoming high edges 
        {
            auto* cur = this->first_high_incoming;
            while(cur != nullptr) {
                //m = std::min(m, cur->m + *variable_cost + *cumulative_sum - *(cur->cumulative_sum));
                m = std::min(m, cur->m + *(cur->variable_cost));
                cur = cur->next_high_incoming;
            }
        }

        //std::cout << "forward step m for " << this << " = " << m << "\n";
        assert(std::isfinite(m));
        // assert(std::abs(m - cost_from_first()) <= 1e-8);

        check_bdd_branch_node(*this);
    }
    */
    template<typename DERIVED>
    void bdd_branch_node_opt_base<DERIVED>::forward_step()
    {
        assert(std::isfinite(this->m));
        if(!this->is_terminal(this->low_outgoing))
            this->low_outgoing->m = std::min(this->low_outgoing->m, this->m);
        if(!this->is_terminal(this->high_outgoing))
            this->high_outgoing->m = std::min(this->high_outgoing->m, this->m + *this->variable_cost);
    }

    template<typename DERIVED>
    void bdd_branch_node_opt_base<DERIVED>::backward_step()
    {
        check_bdd_branch_node(*this);

        const double low_cost = this->low_outgoing->m;
        // low edge
        /*
        const double cost_tmp = [&]() {
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0()) {
                return std::numeric_limits<double>::infinity();
            } else if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1()) {
                return 0.0;
            } else {
                return this->low_outgoing->m;
            }
        }();
        */

        // high edge
        const double high_cost = this->high_outgoing->m + *variable_cost;
        /*
        const double high_cost = [&]() {
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0()) {
                return std::numeric_limits<double>::infinity();
            } else if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1()) {
                return *variable_cost;
            } else {
                return this->high_outgoing->m + *variable_cost;
            }
        }();
        */

        assert(!std::isnan(low_cost));
        assert(!std::isnan(high_cost));
        assert(std::isfinite(std::min(low_cost,high_cost)));
        m = std::min(low_cost, high_cost);

        check_bdd_branch_node(*this);
        // assert(std::abs(m - cost_from_terminal()) <= 1e-8);
    }

    template<typename DERIVED>
    double bdd_branch_node_opt_base<DERIVED>::cost_from_first() const
    {
        assert(false);
        return 0.0;
        // TODO: move to opt_base with incoming pointers
        /*
        // TODO: only works if no bdd nodes skips variables
        double c = std::numeric_limits<double>::infinity();
        if(this->is_first())
            return 0.0;
        
        // iterate over all incoming low edges 
        {
            auto* cur = this->first_low_incoming;
            while(cur != nullptr) {
                c = std::min(c, cur->cost_from_first());
                cur = cur->next_low_incoming;
            }
        }

        // iterate over all incoming high edges 
        {
            auto* cur = this->first_high_incoming;
            while(cur != nullptr) {
                c = std::min(c, cur->cost_from_first() + *cur->variable_cost);
                cur = cur->next_high_incoming;
            }
        }

        return c;
        */
    }

    template<typename DERIVED>
    double bdd_branch_node_opt_base<DERIVED>::cost_from_terminal() const
    {
        // TODO: only works if no bdd nodes skips variables
        // low edge
        const double low_cost = [&]() {
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0()) {
                return std::numeric_limits<double>::infinity();
            } else if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1()) {
                return 0.0;
            } else {
                return this->low_outgoing->cost_from_terminal();;
            }
        }();

        // high edge
        const double high_cost = [&]() {
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0()) {
                return std::numeric_limits<double>::infinity(); 
            } else if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1()) {
                return *variable_cost; 
            } else {
                return this->high_outgoing->cost_from_terminal() + *variable_cost;
            }
        }();

        return std::min(low_cost, high_cost); 
    }

    template<typename DERIVED>
    std::array<double,2> bdd_branch_node_opt_base<DERIVED>::min_marginal() const
    {
        check_bdd_branch_node(*this);

        // assert(std::abs(m - cost_from_first()) <= 1e-8);
        if(!bdd_branch_node_opt_base<DERIVED>::is_terminal(this->low_outgoing)) {
            // assert(std::abs(low_outgoing->m - low_outgoing->cost_from_terminal()) <= 1e-8);
        }
        if(!bdd_branch_node_opt_base<DERIVED>::is_terminal(this->high_outgoing)) {
            // assert(std::abs(high_outgoing->m - high_outgoing->cost_from_terminal()) <= 1e-8);
        }

        const double m0 = this->m + this->low_outgoing->m;
        /*
        const double m0 = [&]() {
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0())
                return std::numeric_limits<double>::infinity();
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1())
                return this->m;
            return this->m + this->low_outgoing->m;
        }();
        */

        const double m1 = this->m + *this->variable_cost + this->high_outgoing->m;
        /*
        const double m1 = [&]() {
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0())
                return std::numeric_limits<double>::infinity();
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1())
                return this->m + *this->variable_cost;
            return this->m + *this->variable_cost + this->high_outgoing->m;
        }();
        */

        assert(std::isfinite(std::min(m0,m1)));

        return {m0,m1};
    }

    template<typename DERIVED>
    std::array<double,2> bdd_branch_node_opt_base<DERIVED>::min_marginal_debug() const
    {
        check_bdd_branch_node(*this);

        const double m_debug = cost_from_first();

        const double m0 = [&]() {
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0())
                return std::numeric_limits<double>::infinity();
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1())
                return m_debug;
            return m_debug + this->low_outgoing->cost_from_terminal();
        }();

        const double m1 = [&]() {
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0())
                return std::numeric_limits<double>::infinity();
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1())
                return m_debug + *this->variable_cost;
            return m_debug + *this->variable_cost + this->high_outgoing->cost_from_terminal();
        }();

        assert(std::isfinite(std::min(m0,m1)));

        return {m0,m1}; 
    }

    // TODO: below static initializer should be in ...opt_base class. I have not figured out yet how to have it invoked there properly.
    class bdd_branch_node_opt : public bdd_branch_node_opt_base<bdd_branch_node_opt> {
        private:
            static inline double terminal_0_cost = std::numeric_limits<double>::infinity();
            static inline double terminal_1_cost = 0.0;
        public:
            struct static_constructor {
                static_constructor() {
                    bdd_branch_node_opt::terminal_0()->m = std::numeric_limits<double>::infinity();
                    bdd_branch_node_opt::terminal_0()->variable_cost = &terminal_0_cost;

                    bdd_branch_node_opt::terminal_1()->m = 0.0;
                    bdd_branch_node_opt::terminal_1()->variable_cost = &terminal_1_cost;
                }
            };
            static inline static_constructor static_constructor_; 
    };

    //////////////////////////////////////
    // Smoothed Optimization Branch Node
    //////////////////////////////////////

    struct bdd_branch_node_exp_sum_entry
    {
        std::array<double, 2> sum;
        std::array<double, 2> max;
    };

    // TODO: C++20: make default operator==
    inline bool operator==(const bdd_branch_node_exp_sum_entry &x, const bdd_branch_node_exp_sum_entry &y) 
    { 
        return x.sum == y.sum && x.max == y.max;
    }

    /*
    struct bdd_min_marginal_averaging_smoothed_options
    {
        double cost_scaling_ = 1.0;
    };

    template<typename DERIVED>
    class bdd_branch_node_opt_smoothed_base : public bdd_branch_node_opt_base<DERIVED>
    {
    public:
        // below two are provided by base
        //double *variable_cost = nullptr;
        //double m = 0.0;

        double current_max = 0.0; // intermediate maximum value in the exp sum, used for stabilizing log-sum-exp computation. Also referred to as streamed log sum exp.

        // From C++20
        friend bool operator==(const bdd_branch_node_opt_smoothed_base &x, const bdd_branch_node_opt_smoothed_base &y);

        void smooth_backward_step();
        void smooth_forward_step();

        // Debug functions for checking correctness of forward and backward step
        double smooth_cost_from_first() const;
        double smooth_cost_from_terminal() const;

        bdd_branch_node_exp_sum_entry exp_sums() const;
        template <typename BDD_BRANCH_NODE_ITERATOR>
        static bdd_branch_node_exp_sum_entry exp_sums(BDD_BRANCH_NODE_ITERATOR bdd_node_begin, BDD_BRANCH_NODE_ITERATOR bdd_node_end);
    };

    class bdd_branch_node_opt_smoothed : public bdd_branch_node_opt_smoothed_base<bdd_branch_node_opt_smoothed>
    {};

    template<typename DERIVED>
    void bdd_branch_node_opt_smoothed_base<DERIVED>::smooth_forward_step()
    {
        check_bdd_branch_node(*this);

        if (this->is_first()) {
            this->m = 1.0; // == exp(0);
            current_max = 0.0;
            return;
        }

        this->m = 0.0; 
        current_max = -std::numeric_limits<double>::infinity();

        // iterate over all incoming low edges
        {
            for(auto *cur = this->first_low_incoming; cur != nullptr; cur = cur->next_low_incoming)
            {
                if(cur->current_max < current_max)
                {
                    this->m += std::exp(cur->current_max - current_max) * cur->m;
                }
                else
                {
                    this->m *= std::exp(current_max - cur->current_max);
                    this->m += cur->m;
                    current_max = cur->current_max;
                }
                assert(std::isfinite(this->m));
            }
        }

        // iterate over all incoming high edges
        {
            for(auto *cur = this->first_high_incoming; cur != nullptr; cur = cur->next_high_incoming)
            {
                if(cur->current_max -*cur->variable_cost < current_max)
                {
                    this->m += std::exp((cur->current_max -*cur->variable_cost) - current_max) * cur->m;
                } 
                else
                {
                    this->m *= std::exp(current_max - (cur->current_max - *cur->variable_cost));
                    this->m += cur->m; //1.0;
                    current_max = cur->current_max - *cur->variable_cost;
                }
                //m += std::exp(-*(cur->variable_cost)) * cur->m;
                assert(std::isfinite(this->m));
            }
        }

        assert(std::isfinite(this->m));
        assert(this->m > 0.0);
        assert(this->m < 10000.0);
        check_bdd_branch_node(*this);
    }

    template<typename DERIVED>
    void bdd_branch_node_opt_smoothed_base<DERIVED>::smooth_backward_step()
    {
        check_bdd_branch_node(*this);

        // low edge
        const auto [low_cost, low_max] = [&]() -> std::array<double,2> {
            if (this->low_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0())
                return {0.0, -std::numeric_limits<double>::infinity()};
            else if (this->low_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
                return {std::exp(0.0), 0.0};
            else
                return {this->low_outgoing->m, this->low_outgoing->current_max};
        }();

        // high edge
        const auto [high_cost, high_max] = [&]() -> std::array<double,2> {
            //if (high_outgoing == bdd_branch_node_opt_smoothed::terminal_0())
            //    return {0.0, -std::numeric_limits<double>::infinity()};
            //else if (high_outgoing == bdd_branch_node_opt_smoothed::terminal_1())
            //    return {std::exp(-*variable_cost - low_max)), -*variable_cost};
            //else
            //    return {std::exp(-*variable_cost) * high_outgoing->m, -*variable_cost + high_outgoing->current_max};
            if (this->high_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0())
                return {0.0, -std::numeric_limits<double>::infinity()};
            else if (this->high_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
                return {std::exp(0.0), 0.0};
            else
                return {this->high_outgoing->m, this->high_outgoing->current_max};
        }();

        assert(std::isfinite(low_cost));
        assert(std::isfinite(high_cost));
        this->m = low_cost;
        current_max = low_max;
        //current_max = 0;
        //m += std::exp(-*variable_cost)*high_cost;
        //return;

        if (std::isfinite(high_max))
        {
            if (high_max - *this->variable_cost < low_max)
            {
                this->m += std::exp((high_max - *this->variable_cost) - current_max) * high_cost;
                //m += std::exp(-*variable_cost - current_max) * high_cost;
            }
            else
            {
                this->m *= std::exp(current_max - (high_max - *this->variable_cost));
                this->m += high_cost;//1.0;
                current_max = high_max - *this->variable_cost;
            }
        }
        //m = low_cost + high_cost;

        assert(std::isfinite(this->m));
        assert(std::abs(this->m) < 10000.0);
        assert(std::isfinite(current_max));

        check_bdd_branch_node(*this);
        //assert(std::abs(m - cost_from_terminal()) <= 1e-8);
    }

    template<typename DERIVED>
    double bdd_branch_node_opt_smoothed_base<DERIVED>::smooth_cost_from_first() const
    {
        double c = 0.0;

        if (this->is_first())
            return 0.0;

        // iterate over all incoming low edges
        for (auto *cur = this->first_low_incoming; cur != nullptr; cur = cur->next_low_incoming)
            c += cur->smooth_cost_from_first();

        // iterate over all incoming high edges
        for (auto *cur = this->first_high_incoming; cur != nullptr; cur = cur->next_high_incoming)
            c += std::exp(-*cur->variable_cost) * cur->smooth_cost_from_first(); // ??

        return c;
    }

    template<typename DERIVED>
    double bdd_branch_node_opt_smoothed_base<DERIVED>::smooth_cost_from_terminal() const
    {
        // TODO: only works if no bdd nodes skips variables
        // low edge
        const double low_cost = [&]() {
            if (this->low_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0())
                return 0.0;
            else if (this->low_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
                return 1.0;
            else
                return this->low_outgoing->smooth_cost_from_terminal();
        }();

        // high edge
        const double high_cost = [&]() {
            if (this->high_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0())
                return 0.0;
            else if (this->high_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
                return std::exp(-*this->variable_cost);
            else
                return this->high_outgoing->smooth_cost_from_terminal() + std::exp(-*this->variable_cost);
        }();

        return low_cost + high_cost;
    }

    template<typename DERIVED>
    bdd_branch_node_exp_sum_entry bdd_branch_node_opt_smoothed_base<DERIVED>::exp_sums() const
    {
        check_bdd_branch_node(*this);

        // assert(std::abs(m - cost_from_first()) <= 1e-8);
        if (!bdd_branch_node_opt_smoothed_base<DERIVED>::is_terminal(this->low_outgoing))
        {
            //assert(std::abs(low_outgoing->m - low_outgoing->cost_from_terminal()) <= 1e-8);
        }
        if (!bdd_branch_node_opt_smoothed_base<DERIVED>::is_terminal(this->high_outgoing))
        {
            //assert(std::abs(high_outgoing->m - high_outgoing->cost_from_terminal()) <= 1e-8);
        }

        bdd_branch_node_exp_sum_entry e;

        std::tie(e.sum[0], e.max[0]) = [&]() -> std::tuple<double, double> {
            if (this->low_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0())
                return {0.0, -std::numeric_limits<double>::infinity()};
            if (this->low_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
                return {this->m, current_max};
            else
                return {this->m * this->low_outgoing->m, current_max + this->low_outgoing->current_max};
        }();

        std::tie(e.sum[1], e.max[1]) = [&]() -> std::tuple<double, double> {
            if (this->high_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0())
                return {0.0, -std::numeric_limits<double>::infinity()};
            if (this->high_outgoing == bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
            {
                //const double new_max = std::max(this->current_max, this->current_max - *this->variable_cost);
                //return {this->m * std::exp(-*this->variable_cost + this->current_max - new_max), new_max};
                return {this->m, this->current_max - *this->variable_cost};
            }
            else
            {
                //const double new_max = std::max({this->current_max, this->current_max - *this->variable_cost, this->high_outgoing->current_max});
                //return {this->m * std::exp(-*this->variable_cost + this->current_max + this->high_outgoing->current_max - new_max) * this->high_outgoing->m, new_max};
                return {this->m * this->high_outgoing->m, this->current_max - *this->variable_cost + this->high_outgoing->current_max};
            }
        }();

        assert(std::isfinite(e.sum[0]));
        assert(std::isfinite(e.sum[1]));
        assert(e.sum[0] >= 0.0);
        assert(e.sum[1] >= 0.0);
        assert(e.sum[0] > 0 || e.sum[1] > 0);
        if(this->low_outgoing != bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0() && this->low_outgoing != bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
        {
            assert(e.sum[0] > 0);
            assert(std::isfinite(e.max[0]));
        }
        if (this->high_outgoing != bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_0() && this->high_outgoing != bdd_branch_node_opt_smoothed_base<DERIVED>::terminal_1())
        {
            assert(e.sum[1] > 0);
            assert(std::isfinite(e.max[1]));
        }

        assert(std::abs(e.sum[0]) < 10000.0);
        assert(std::abs(e.sum[1]) < 10000.0);

        return e;
    }

template<typename DERIVED>
template<typename BDD_BRANCH_NODE_ITERATOR>
bdd_branch_node_exp_sum_entry bdd_branch_node_opt_smoothed_base<DERIVED>::exp_sums(BDD_BRANCH_NODE_ITERATOR bdd_node_begin, BDD_BRANCH_NODE_ITERATOR bdd_node_end)
{
    bdd_branch_node_exp_sum_entry e;
    e.sum = {0.0, 0.0};
    e.max = {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    std::array<double, 2> s = {0.0, 0.0};
    std::array<double, 2> current_max = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    for(auto bdd_node_it = bdd_node_begin; bdd_node_it != bdd_node_end; ++bdd_node_it)
    {
        const auto &bdd = *bdd_node_it;
        const auto bdd_exp_sums = bdd.exp_sums();
        //std::cout << "var" << var << ", bdd index = " << bdd_index << ": " << "\n";
        //std::cout << bdd_exp_sums.sum[0] << "," << bdd_exp_sums.sum[1] << ";" << bdd_exp_sums.max[0] << "," << bdd_exp_sums.max[1] << "\n";

        if (bdd_exp_sums.sum[0] > 0)
        {
            if (current_max[0] > bdd_exp_sums.max[0])
            {
                s[0] += bdd_exp_sums.sum[0] * std::exp(bdd_exp_sums.max[0] - current_max[0]);
            }
            else
            {
                s[0] *= std::exp(current_max[0] - bdd_exp_sums.max[0]);
                s[0] += bdd_exp_sums.sum[0];
                current_max[0] = bdd_exp_sums.max[0];
            }
        }

        if (bdd_exp_sums.sum[1] > 0)
        {
            if (current_max[1] > bdd_exp_sums.max[1])
            {
                s[1] += bdd_exp_sums.sum[1] * std::exp(bdd_exp_sums.max[1] - current_max[1]);
            }
            else
            {
                s[1] *= std::exp(current_max[1] - bdd_exp_sums.max[1]);
                s[1] += bdd_exp_sums.sum[1];
                current_max[1] = bdd_exp_sums.max[1];
            }
        }
        assert(std::isfinite(s[0]));
        assert(std::isfinite(s[1]));
    }
    //std::cout << s[0] << "," << s[1] << ";" << current_max[0] << "," << current_max[1] << "\n";
    assert(s[0] > 0);
    assert(s[1] > 0);
    return {s, current_max};
}
*/

    /*
    std::array<double, 2> bdd_branch_node_opt_smoothed::min_marginal_debug() const
    {
        check_bdd_branch_node(*this);

        const double m_debug = cost_from_first();

        const double m0 = [&]() {
            if (low_outgoing == bdd_branch_node_opt_smoothed::terminal_0())
                return std::numeric_limits<double>::infinity();
            if (low_outgoing == bdd_branch_node_opt_smoothed::terminal_1())
                return m_debug;
            return m_debug + this->low_outgoing->cost_from_terminal();
        }();

        const double m1 = [&]() {
            if (high_outgoing == bdd_branch_node_opt_smoothed::terminal_0())
                return std::numeric_limits<double>::infinity();
            if (high_outgoing == bdd_branch_node_opt_smoothed::terminal_1())
                return m_debug + *this->variable_cost;
            return m_debug + *this->variable_cost + this->high_outgoing->cost_from_terminal();
        }();

        assert(std::isfinite(std::min(m0, m1)));

        return {m0, m1};
    }
    */

    /////////////////////////////////
    // Variable Fixing Branch Node
    /////////////////////////////////


    class bdd_branch_node_fix : public bdd_branch_node_incoming_pointers<bdd_branch_node_fix> {
        public:
            bdd_branch_node_fix* prev_low_incoming = nullptr;
            bdd_branch_node_fix* prev_high_incoming = nullptr;

            size_t m = 0; // solution count

            bdd_variable_fix* bdd_var;

            // From C++20
            friend bool operator==(const bdd_branch_node_fix& x, const bdd_branch_node_fix& y);

            void count_forward_step();
            void count_backward_step();

            double count_low();
            double count_high();

            struct static_constructor {
                static_constructor() {
                    bdd_branch_node_fix::terminal_0()->m = 0.0;
                    bdd_branch_node_fix::terminal_1()->m = 1.0;
                }
            };
            static inline static_constructor static_constructor_; 
    };

    inline bool operator==(const bdd_branch_node_fix& x, const bdd_branch_node_fix& y)
    {
        const bool equal = (x.low_outgoing == y.low_outgoing &&
            x.high_outgoing == y.high_outgoing &&
            x.first_low_incoming == y.first_low_incoming &&
            x.first_high_incoming == y.first_high_incoming &&
            x.next_low_incoming == y.next_low_incoming &&
            x.next_high_incoming == y.next_high_incoming &&
            x.prev_low_incoming == y.prev_low_incoming &&
            x.prev_high_incoming == y.prev_high_incoming &&
            x.bdd_var == y.bdd_var);
        return equal;
    }

    inline void bdd_branch_node_fix::count_forward_step()
    {
        // if(this->is_first()) {
        //     m = 1.0;
        //     return;
        // }

        // m = 0.0;

        // // iterate over all incoming low edges 
        // {
        //     auto* cur = this->first_low_incoming;
        //     while(cur != nullptr) {
        //         m += cur->m;
        //         cur = cur->next_low_incoming;
        //     }
        // }

        // // iterate over all incoming high edges 
        // {
        //     auto* cur = this->first_high_incoming;
        //     while(cur != nullptr) {
        //         m += cur->m;
        //         cur = cur->next_high_incoming;
        //     }
        // }

        if (m == 0)
            m = 1;
        if (!this->is_terminal(this->low_outgoing))
            low_outgoing->m += m;
        if (!this->is_terminal(this->high_outgoing))
            high_outgoing->m += m; 
    }

    inline void bdd_branch_node_fix::count_backward_step()
    {
        // // low edge
        // const double low_count = [&]() {
        //     if(this->low_outgoing == bdd_branch_node_fix::terminal_0()) {
        //         return 0.0;
        //     } else if(this->low_outgoing == bdd_branch_node_fix::terminal_1()) {
        //         return 1.0;
        //     } else {
        //         return this->low_outgoing->m;
        //     }
        // }();

        // // high edge
        // const double high_count = [&]() {
        //     if(this->high_outgoing == bdd_branch_node_fix::terminal_0()) {
        //         return 0.0; 
        //     } else if(this->high_outgoing == bdd_branch_node_fix::terminal_1()) {
        //         return 1.0; 
        //     } else {
        //         return this->high_outgoing->m;
        //     }
        // }();

        size_t low_count = this->low_outgoing->m;
        size_t high_count = this->high_outgoing->m;

        m = low_count + high_count;
    }

    inline double bdd_branch_node_fix::count_low()
    {
        if (this->low_outgoing == bdd_branch_node_fix::terminal_0())
            return 0.0;
        else if (this->low_outgoing == bdd_branch_node_fix::terminal_1())
            return m;
        else
            return m * this->low_outgoing->m;
    }

    inline double bdd_branch_node_fix::count_high()
    {
        if (this->high_outgoing == bdd_branch_node_fix::terminal_0())
            return 0.0;
        else if (this->high_outgoing == bdd_branch_node_fix::terminal_1())
            return m;
        else
            return m * this->high_outgoing->m;
    }

    //////////////////////

    // bdd branch node with individual arc costs, for use in decomposition bdd base (since there are Lagrange multipliers for individual arcs)
    template<typename DERIVED>
    class bdd_branch_node_opt_arc_cost_base : public bdd_branch_node<DERIVED> {
        public:
            double low_cost = 0.0;
            double high_cost = 0.0;
            double m = 0.0; // intermediate value of shortest path from either terminal or first node (depending on algorithm state)

            // From C++20
            friend bool operator==(const bdd_branch_node_opt_base<DERIVED>& x, const bdd_branch_node_opt_base<DERIVED>& y);

            void backward_step();
            void forward_step();

            std::array<double,2> min_marginal() const;
    };

    template<typename DERIVED>
void bdd_branch_node_opt_arc_cost_base<DERIVED>::backward_step()
{
        // low edge
        const double low_path_cost = this->low_outgoing->m + low_cost;
        /*
        const double low_path_cost = [&]() {
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0()) {
                return std::numeric_limits<double>::infinity();
            } else if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1()) {
                return low_cost;
            } else {
                return this->low_outgoing->m + low_cost;
            }
        }();
        */

        // high edge
        const double high_path_cost = this->high_outgoing->m + high_cost;
        /*
        const double high_path_cost = [&]() {
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0()) {
                return std::numeric_limits<double>::infinity(); 
            } else if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1()) {
                return high_cost; 
            } else {
                return this->high_outgoing->m + high_cost;
            }
        }();
        */

        assert(!std::isnan(low_path_cost));
        assert(!std::isnan(high_path_cost));
        assert(std::isfinite(std::min(low_path_cost,high_path_cost)));
        m = std::min(low_path_cost, high_path_cost); 
}

    template<typename DERIVED>
    void bdd_branch_node_opt_arc_cost_base<DERIVED>::forward_step()
    {
        check_bdd_branch_node(*this);

        assert(std::isfinite(this->m));
        if(!this->is_terminal(this->low_outgoing))
            this->low_outgoing->m = std::min(this->low_outgoing->m, this->m + this->low_cost);
        else
            assert(this->low_outgoing->m == 0 || this->low_outgoing->m == std::numeric_limits<double>::infinity());
        if(!this->is_terminal(this->high_outgoing))
            this->high_outgoing->m = std::min(this->high_outgoing->m, this->m + this->high_cost);
        else
            assert(this->high_outgoing->m == 0 || this->high_outgoing->m == std::numeric_limits<double>::infinity());
    }

/*
    template<typename DERIVED>
    void bdd_branch_node_opt_arc_cost_base<DERIVED>::forward_step()
    {
        check_bdd_branch_node(*this);

        if(this->is_first()) {
            m = 0.0;
            return;
        }

        m = std::numeric_limits<double>::infinity();

        // iterate over all incoming low edges 
        {
            auto* cur = this->first_low_incoming;
            while(cur != nullptr) {
                m = std::min(m, cur->m + cur->low_cost);
                cur = cur->next_low_incoming;
            }
        }

        // iterate over all incoming high edges 
        {
            auto* cur = this->first_high_incoming;
            while(cur != nullptr) {
                m = std::min(m, cur->m + cur->high_cost);
                cur = cur->next_high_incoming;
            }
        }

        assert(std::isfinite(m));
    }
    */


    template<typename DERIVED>
    std::array<double,2> bdd_branch_node_opt_arc_cost_base<DERIVED>::min_marginal() const
    {
        const double m0 = this->m + this->low_cost + this->low_outgoing->m;
        /*
        const double m0 = [&]() {
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0())
                return std::numeric_limits<double>::infinity();
            if(this->low_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1())
                return this->m + this->low_cost;
            return this->m + this->low_cost + this->low_outgoing->m;
        }();
        */

        const double m1 = this->m + this->high_cost + this->high_outgoing->m;
        /*
        const double m1 = [&]() {
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_0())
                return std::numeric_limits<double>::infinity();
            if(this->high_outgoing == bdd_branch_node_opt_base<DERIVED>::terminal_1())
                return this->m + this->high_cost;
            return this->m + this->high_cost + this->high_outgoing->m;
        }();
        */

        assert(std::isfinite(std::min(m0,m1)));

        return {m0,m1};
    }

    class bdd_branch_node_opt_arc_cost : public bdd_branch_node_opt_arc_cost_base<bdd_branch_node_opt_arc_cost> {
        public:
            struct static_constructor {
                static_constructor() {
                    bdd_branch_node_opt_arc_cost::terminal_0()->m = std::numeric_limits<double>::infinity();
                    bdd_branch_node_opt_arc_cost::terminal_0()->low_cost = std::numeric_limits<double>::infinity();
                    bdd_branch_node_opt_arc_cost::terminal_0()->high_cost = std::numeric_limits<double>::infinity();

                    bdd_branch_node_opt_arc_cost::terminal_1()->m = 0.0;
                    bdd_branch_node_opt_arc_cost::terminal_1()->low_cost = 0.0;
                    bdd_branch_node_opt_arc_cost::terminal_1()->high_cost = 0.0;
                }
            };
            static inline static_constructor static_constructor_; 
    };

}
