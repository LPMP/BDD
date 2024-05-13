#pragma once 

#include <numeric>
#include <array>
#include <limits>
#include <type_traits>
#include <cstdint>
#include <cassert>
#include <cstddef>

namespace LPMP {

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    class bdd_branch_instruction_base {
        public:
            using value_type = REAL;
            using offset_type = OFFSET_TYPE;
            // offsets are added to the address of the current bdd_branch_instruction_base<REAL,DERIVED>. The compute address points to the bdd_branch_node_vec
            REAL m = std::numeric_limits<REAL>::infinity();
            REAL low_cost = 0.0;
            REAL high_cost = 0.0;
            OFFSET_TYPE offset_low = 0;
            OFFSET_TYPE offset_high = 0;

            void prepare_forward_step();
            void forward_step();
            void backward_step();

            std::array<REAL,2> min_marginals() const;

            constexpr static OFFSET_TYPE terminal_0_offset = std::numeric_limits<OFFSET_TYPE>::max();
            constexpr static OFFSET_TYPE terminal_1_offset = std::numeric_limits<OFFSET_TYPE>::max()-1;

            DERIVED* address(const OFFSET_TYPE offset);
            const DERIVED* address(const OFFSET_TYPE offset) const;
            OFFSET_TYPE synthesize_address(const DERIVED* node) const;

            bool node_initialized() const;

            ~bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>()
            {
                static_assert(std::is_same_v<REAL, float> || std::is_same_v<REAL, double>, "REAL must be floating point type");
                static_assert(std::is_integral_v<OFFSET_TYPE> && std::is_unsigned_v<OFFSET_TYPE>, "OFFSET_TYPE must be unsigned integral type");
            }
    };

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    class bdd_branch_instruction_bdd_index_base : public bdd_branch_instruction_base<REAL, OFFSET_TYPE, DERIVED> {
        public: 
            constexpr static uint32_t inactive_bdd_index = std::numeric_limits<uint32_t>::max();
            // for distinguishing in bdd base from which bdd the instruction is
            uint32_t bdd_index = inactive_bdd_index;

            void min_marginal(std::array<REAL,2>* reduced_min_marginals) const;
            void set_marginal(std::array<REAL,2>* min_marginals, const std::array<REAL,2> avg_marginals);


            bool node_initialized() const;
    };

    template<typename REAL, typename OFFSET_TYPE>
    class bdd_branch_instruction : public bdd_branch_instruction_base<REAL, OFFSET_TYPE, bdd_branch_instruction<REAL,OFFSET_TYPE>> {};

    // with bdd index
    template<typename REAL, typename OFFSET_TYPE>
    class bdd_branch_instruction_bdd_index : public bdd_branch_instruction_bdd_index_base<REAL,OFFSET_TYPE,bdd_branch_instruction_bdd_index<REAL,OFFSET_TYPE>> { };

    ////////////////////
    // implementation //
    ////////////////////

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    DERIVED* bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::address(const OFFSET_TYPE offset)
    {
        assert(offset_low > 0 && offset_high > 0);
        assert(offset != terminal_0_offset && offset != terminal_1_offset);
        return static_cast<DERIVED*>(this) + offset;
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    const DERIVED* bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::address(const OFFSET_TYPE offset) const
    {
        assert(offset_low > 0 && offset_high > 0);
        assert(offset != terminal_0_offset && offset != terminal_1_offset);
        return static_cast<const DERIVED*>(this) + offset;
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    OFFSET_TYPE bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::synthesize_address(const DERIVED* node) const
    {
        assert(static_cast<const DERIVED*>(this) < node);
        assert(std::distance(static_cast<const DERIVED*>(this), node) < std::numeric_limits<OFFSET_TYPE>::max());
        assert(std::distance(static_cast<const DERIVED*>(this), node) > 0);
        return std::distance(static_cast<const DERIVED*>(this), node);
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    void bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::backward_step()
    {
        //if(offset_low == terminal_0_offset)
        //    assert(low_cost == std::numeric_limits<REAL>::infinity());
        //if(offset_high == terminal_0_offset)
        //    assert(high_cost == std::numeric_limits<REAL>::infinity());

        if(offset_low == terminal_0_offset || offset_low == terminal_1_offset)
        {
            m = low_cost;
        }
        else
        {
            const auto low_branch_node = address(offset_low);
            assert(!std::isnan(low_branch_node->m));
            m = low_branch_node->m + low_cost;
        }

        if(offset_high == terminal_0_offset || offset_high == terminal_1_offset)
        {
            m = std::min(m, high_cost);
        }
        else
        {
            const auto high_branch_node = address(offset_high);
            assert(!std::isnan(high_branch_node->m));
            m = std::min(m, high_branch_node->m + high_cost);
        }

        assert(!std::isnan(m));
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    void bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::prepare_forward_step()
    {
        assert(offset_low > 0 && offset_high > 0);
        if(offset_low != terminal_0_offset && offset_low != terminal_1_offset)
        {
            const auto low_branch_node = address(offset_low);
            low_branch_node->m = std::numeric_limits<REAL>::infinity(); 
        }

        if(offset_high != terminal_0_offset && offset_high != terminal_1_offset)
        {
            const auto high_branch_node = address(offset_high);
            high_branch_node->m = std::numeric_limits<REAL>::infinity();
        }
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    void bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::forward_step()
    {
        assert(offset_low > 0 && offset_high > 0);
        if(offset_low != terminal_0_offset && offset_low != terminal_1_offset)
        {
            const auto low_branch_node = address(offset_low);
            assert(!std::isnan(low_branch_node->m));
            low_branch_node->m = std::min(low_branch_node->m, m + low_cost);
            assert(!std::isnan(low_branch_node->m));
        }

        if(offset_high != terminal_0_offset && offset_high != terminal_1_offset)
        {
            const auto high_branch_node = address(offset_high);
            assert(!std::isnan(high_branch_node->m));
            high_branch_node->m = std::min(high_branch_node->m, m + high_cost);
            assert(!std::isnan(high_branch_node->m));
        }
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    std::array<REAL,2> bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::min_marginals() const
    {
        assert(offset_low > 0 && offset_high > 0);
        std::array<REAL,2> mm;
        if(offset_low == terminal_0_offset || offset_low == terminal_1_offset)
        {
            mm[0] = m + low_cost;
            assert(!std::isnan(mm[0]));
        }
        else
        {
            const auto low_branch_node = address(offset_low);
            mm[0] = m + low_cost + low_branch_node->m;
            assert(!std::isnan(mm[0]));
        }

        if(offset_high == terminal_0_offset || offset_high == terminal_1_offset)
        {
            mm[1] = m + high_cost;
            assert(!std::isnan(mm[1]));
        }
        else
        {
            const auto high_branch_node = address(offset_high);
            mm[1] = m + high_cost + high_branch_node->m;
            assert(!std::isnan(mm[1]));
        }

        return mm;
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    void bdd_branch_instruction_bdd_index_base<REAL,OFFSET_TYPE,DERIVED>::min_marginal(std::array<REAL,2>* reduced_min_marginals) const
    {
        assert(this->offset_low > 0 && this->offset_high > 0);
        assert(bdd_index != inactive_bdd_index);
        const auto mm = this->min_marginals();
        reduced_min_marginals[bdd_index][0] = std::min(mm[0], reduced_min_marginals[bdd_index][0]);
        reduced_min_marginals[bdd_index][1] = std::min(mm[1], reduced_min_marginals[bdd_index][1]);
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
    void bdd_branch_instruction_bdd_index_base<REAL,OFFSET_TYPE,DERIVED>::set_marginal(std::array<REAL,2>* reduced_min_marginals, const std::array<REAL,2> avg_marginals)
    {
        assert(!std::isnan(avg_marginals[0]));
        assert(!std::isnan(avg_marginals[1]));
        assert(!std::isnan(reduced_min_marginals[bdd_index][0]));
        assert(!std::isnan(reduced_min_marginals[bdd_index][1]));

        if(std::isfinite(reduced_min_marginals[bdd_index][0]))
            this->low_cost += -reduced_min_marginals[bdd_index][0] + avg_marginals[0];
        else
            this->low_cost = std::numeric_limits<float>::infinity();
        if(std::isfinite(reduced_min_marginals[bdd_index][1]))
            this->high_cost += -reduced_min_marginals[bdd_index][1] + avg_marginals[1];
        else
            this->high_cost = std::numeric_limits<float>::infinity();

        assert(!std::isnan(this->low_cost));
        assert(!std::isnan(this->high_cost));
    }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
        bool bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::node_initialized() const
        {
            if(offset_low == 0 || offset_high == 0) 
                return false;
            return true;
        }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED>
        bool bdd_branch_instruction_bdd_index_base<REAL,OFFSET_TYPE,DERIVED>::node_initialized() const
        {
            if(bdd_branch_instruction_base<REAL,OFFSET_TYPE,DERIVED>::node_initialized() == false)
                return false;
            if(bdd_index == inactive_bdd_index)
                return false;
            return true;
        }

}
