#pragma once 

#include <array>
#include <limits>

namespace LPMP {

    template<typename REAL>
    class bdd_branch_instruction {
        public:
            using value_type = REAL;
            // offsets are added to the address of the current bdd_branch_instruction<REAL>. The compute address points to the bdd_branch_node_vec
            uint32_t offset_low = 0;
            uint32_t offset_high = 0;
            REAL m = std::numeric_limits<REAL>::infinity();
            REAL low_cost = 0.0;
            REAL high_cost = 0.0;

            void prepare_forward_step();
            void forward_step();
            void backward_step();

            std::array<REAL,2> min_marginals();

            constexpr static uint32_t terminal_0_offset = std::numeric_limits<uint32_t>::max();
            constexpr static uint32_t terminal_1_offset = std::numeric_limits<uint32_t>::max()-1;

            bdd_branch_instruction<REAL>* address(uint32_t offset);
            uint32_t synthesize_address(bdd_branch_instruction<REAL>* node);

            ~bdd_branch_instruction<REAL>()
            {
                static_assert(std::is_same_v<REAL, float> || std::is_same_v<REAL, double>, "REAL must be floating point type");
                static_assert(sizeof(uint32_t) == 4, "uint32_t must be quadword");
            }
    };

    // with bdd index
    template<typename REAL>
    class bdd_branch_instruction_bdd_index : public bdd_branch_instruction<REAL> {
        public: 
            constexpr static uint32_t inactive_bdd_index = std::numeric_limits<uint32_t>::max();
            uint32_t bdd_index = inactive_bdd_index;

            void min_marginal(std::array<REAL,2>* reduced_min_marginals);
            void set_marginal(std::array<REAL,2>* min_marginals, const std::array<REAL,2> avg_marginals);

    };

    ////////////////////
    // implementation //
    ////////////////////

    template<typename REAL>
    bdd_branch_instruction<REAL>* bdd_branch_instruction<REAL>::address(uint32_t offset)
    {
        assert(offset != terminal_0_offset && offset != terminal_1_offset);
        return this + offset;
    }

    template<typename REAL>
    uint32_t bdd_branch_instruction<REAL>::synthesize_address(bdd_branch_instruction<REAL>* node)
    {
        assert(this < node);
        assert(std::distance(this, node) < std::numeric_limits<uint32_t>::max());
        assert(std::distance(this, node) > 0);
        return std::distance(this, node);
    }

    template<typename REAL>
    void bdd_branch_instruction<REAL>::backward_step()
    {
        if(offset_low == terminal_0_offset)
            assert(low_cost == std::numeric_limits<REAL>::infinity());

        if(offset_low == terminal_0_offset || offset_low == terminal_1_offset)
        {
            m = low_cost;
        }
        else
        {
            const auto low_branch_node = address(offset_low);
            m = low_branch_node->m + low_cost;
        }

        if(offset_high == terminal_0_offset)
            assert(high_cost == std::numeric_limits<REAL>::infinity());

        if(offset_high == terminal_0_offset || offset_high == terminal_1_offset)
        {
            m = std::min(m, high_cost);
        }
        else
        {
            const auto high_branch_node = address(offset_high);
            m = std::min(m, high_branch_node->m + high_cost);
        }

        assert(std::isfinite(m));
    }

    template<typename REAL>
    void bdd_branch_instruction<REAL>::prepare_forward_step()
    {
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

    template<typename REAL>
    void bdd_branch_instruction<REAL>::forward_step()
    {
        if(offset_low != terminal_0_offset && offset_low != terminal_1_offset)
        {
            const auto low_branch_node = address(offset_low);
            low_branch_node->m = std::min(low_branch_node->m, m + low_cost);
        }

        if(offset_high != terminal_0_offset && offset_high != terminal_1_offset)
        {
            const auto high_branch_node = address(offset_high);
            high_branch_node->m = std::min(high_branch_node->m, m + high_cost);
        }
    }

    template<typename REAL>
    std::array<REAL,2> bdd_branch_instruction<REAL>::min_marginals()
    {
        std::array<REAL,2> mm;
        if(offset_low == terminal_0_offset || offset_low == terminal_1_offset)
        {
            mm[0] = m + low_cost;
        }
        else
        {
            const auto low_branch_node = address(offset_low);
            mm[0] = m + low_cost + low_branch_node->m;
        }

        if(offset_high == terminal_0_offset || offset_high == terminal_1_offset)
        {
            mm[1] = m + high_cost;
        }
        else
        {
            const auto high_branch_node = address(offset_high);
            mm[1] = m + high_cost + high_branch_node->m;
        }

        assert(std::isfinite(std::min(mm[0],mm[1])));
        return mm;
    }

    template<typename REAL>
    void bdd_branch_instruction_bdd_index<REAL>::min_marginal(std::array<REAL,2>* reduced_min_marginals)
    {
        // TODO: use above min marginal
        const auto mm = this->min_marginal();
        reduced_min_marginals[bdd_index][0] = std::min(mm[0], reduced_min_marginals[bdd_index][0]);
        reduced_min_marginals[bdd_index][1] = std::min(mm[1], reduced_min_marginals[bdd_index][1]);
        return;
        if(this->offset_low == this->terminal_0_offset || this->offset_low == this->terminal_1_offset)
        {
            reduced_min_marginals[bdd_index][0] = std::min(this->m + this->low_cost, reduced_min_marginals[bdd_index][0]);
        }
        else
        {
            const auto low_branch_node = this->address(this->offset_low);
            reduced_min_marginals[bdd_index][0] = std::min(this->m + this->low_cost + low_branch_node->m, reduced_min_marginals[bdd_index][0]);
        }

        if(this->offset_high == this->terminal_0_offset || this->offset_high == this->terminal_1_offset)
        {
            reduced_min_marginals[bdd_index][1] = std::min(this->m + this->high_cost, reduced_min_marginals[bdd_index][1]);
        }
        else
        {
            const auto high_branch_node = this->address(this->offset_high);
            reduced_min_marginals[bdd_index][1] = std::min(this->m + this->high_cost + high_branch_node->m, reduced_min_marginals[bdd_index][1]);
        }
    }

    template<typename REAL>
    void bdd_branch_instruction_bdd_index<REAL>::set_marginal(std::array<REAL,2>* reduced_min_marginals, const std::array<REAL,2> avg_marginals)
    {
        assert(std::isfinite(avg_marginals[0]));
        assert(std::isfinite(avg_marginals[1]));
        assert(std::isfinite(reduced_min_marginals[bdd_index][0]));
        this->low_cost += -reduced_min_marginals[bdd_index][0] + avg_marginals[0];
        assert(std::isfinite(reduced_min_marginals[bdd_index][1]));
        this->high_cost += -reduced_min_marginals[bdd_index][1] + avg_marginals[1]; 
    }

}
