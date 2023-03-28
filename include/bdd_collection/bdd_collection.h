#pragma once

#include "bdd_manager/bdd_mgr.h"
#include <vector>
#include <iterator>
#include <unordered_map> // TODO: replace with faster hash map
#include <iterator>
#include <iostream> // TODO: remove

namespace BDD {

    struct bdd_instruction {
        size_t lo = temp_undefined_index;
        size_t hi = temp_undefined_index;
        size_t index = temp_undefined_index;

        constexpr static  size_t botsink_index = std::numeric_limits<size_t>::max()-1;
        bool is_botsink() const { return index == botsink_index; }
        static bdd_instruction botsink() { return {botsink_index, botsink_index, botsink_index}; }

        constexpr static  size_t topsink_index = std::numeric_limits<size_t>::max();
        bool is_topsink() const { return index == topsink_index; }
        static bdd_instruction topsink() { return {topsink_index, topsink_index, topsink_index}; }

        bool is_terminal() const { return is_botsink() || is_topsink(); }

        bool operator==(const bdd_instruction& o) const { return lo == o.lo && hi == o.hi && index == o.index; }
        bool operator!=(const bdd_instruction& o) const { return !(*this == o); }

        // temporary values for building up
        constexpr static size_t temp_botsink_index = std::numeric_limits<size_t>::max();
        constexpr static size_t temp_topsink_index = std::numeric_limits<size_t>::max()-1;
        constexpr static size_t temp_undefined_index = std::numeric_limits<size_t>::max()-2;
    };

    struct bdd_instruction_hasher {
        size_t operator()(const bdd_instruction& bdd) const { return std::hash<size_t>()(bdd.lo) ^ std::hash<size_t>()(bdd.hi) ^ std::hash<size_t>()(bdd.index); }
    };

    template<size_t N>
    struct array_hasher {
        size_t hash_combine(size_t lhs, size_t rhs) const
        {
            lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
            return lhs;
        }
           

        size_t operator()(const std::array<size_t,N>& a) const 
        {
             size_t h = std::hash<size_t>()(a[0]);
             for(std::size_t i=1; i<N; ++i) {
                 h = hash_combine(h, std::hash<size_t>()(a[i]));
             }   
             return h; 
        } 
    };

    class bdd_collection_node;
    // convenience class to make bdd_collection bdds behave similary to node_ref
    class bdd_collection_entry 
    {
        friend class bdd_collection_node;
        public:
            bdd_collection_entry(const size_t _bdd_nr, bdd_collection& _bdd_col);
            std::vector<size_t> variables();
            bdd_collection_entry operator&(bdd_collection_entry& o);
            bdd_collection_node operator[](const size_t i) const;
            size_t nr_nodes() const;
            size_t nr_nodes(const size_t variable) const;
            template<typename ITERATOR>
                void rebase(ITERATOR var_map_begin, ITERATOR var_map_end);
            template<typename VAR_MAP>
                void rebase(const VAR_MAP& var_map);
            std::vector<size_t> rebase_to_contiguous();

            bdd_collection_node root_node() const;
            bdd_collection_node first_node_postorder() const;
            bdd_collection_node botsink() const;
            bdd_collection_node topsink() const;

        private:
            const size_t bdd_nr;
            bdd_collection& bdd_col;
    };

    // convenience class for wrapping bdd node
    class bdd_collection_node
    {
        friend class std::hash<bdd_collection_node>;
        public:
            bdd_collection_node(const size_t _i, const bdd_collection_entry _bce);
            bdd_collection_node lo() const;
            bdd_collection_node low() const { return lo(); }
            bdd_collection_node hi() const;
            bdd_collection_node high() const { return hi(); }
            size_t variable() const;
            bool is_botsink() const;
            bool is_topsink() const;
            bool is_terminal() const;
            bdd_collection_node next_postorder() const;
            bool operator==(const bdd_collection_node& o) const;
            bool operator!=(const bdd_collection_node& o) const;
            bdd_collection_node operator=(const bdd_collection_node& o);

            // for rerouting
            void set_lo_arc(bdd_collection_node& node);
            void set_hi_arc(bdd_collection_node& node);
            void set_hi_to_0_terminal();
            void set_hi_to_1_terminal();
            void set_lo_to_0_terminal();
            void set_lo_to_1_terminal();

        private:
            size_t i; 
            bdd_collection_entry bce;
    };


    class bdd_collection {
        friend class bdd_collection_node;
        friend class bdd_collection_entry;
        public:
            // synthesize bdd unless it has too many nodes. If so, return std::numeric_limits<size_t>::max()
            size_t bdd_and(const size_t i, const size_t j, bdd_collection& o);
            size_t bdd_and(const size_t i, const size_t j);
            size_t bdd_and(const int i, const int j, bdd_collection& o);
            size_t bdd_and(const int i, const int j);

            template<typename BDD_ITERATOR>
                size_t bdd_and(BDD_ITERATOR bdd_begin, BDD_ITERATOR bdd_end, bdd_collection& o);
            template<typename BDD_ITERATOR>
                size_t bdd_and(BDD_ITERATOR bdd_begin, BDD_ITERATOR bdd_end);

            // compute disjunction 
            size_t bdd_or(const size_t i, const size_t j);
            template<typename VAR_MAP>
                size_t bdd_or_var(const size_t i, const VAR_MAP& positive_variables, const VAR_MAP& negative_variables);

            // compute how many solutions there are after forcing some variables to be true or false
            template<typename VAR_MAP>
                size_t bdd_nr_solutions(const size_t bdd_nr, const VAR_MAP& positive_variables, const VAR_MAP& negative_variables);

            size_t add_bdd(node_ref bdd);
            node_ref export_bdd(bdd_mgr& mgr, const size_t bdd_nr) const;
            size_t nr_bdds() const { return bdd_delimiters.size()-1; }
            size_t size() const { return nr_bdds(); }
            size_t nr_bdd_nodes(const size_t bdd_nr) const;
            size_t nr_bdd_nodes(const size_t bdd_nr, const size_t variable) const;

            bdd_instruction* begin(const size_t bdd_nr);
            bdd_instruction* end(const size_t bdd_nr);
            const bdd_instruction* cbegin(const size_t bdd_nr) const;
            const bdd_instruction* cend(const size_t bdd_nr) const;
            std::reverse_iterator<bdd_instruction*> rbegin(const size_t bdd_nr);
            std::reverse_iterator<bdd_instruction*> rend(const size_t bdd_nr);

            size_t botsink_index(const size_t bdd_nr) const;
            size_t topsink_index(const size_t bdd_nr) const;

            size_t offset(const bdd_instruction& instr) const;
            size_t offset(const size_t bdd_nr) const { assert(bdd_nr < nr_bdds()); return bdd_delimiters[bdd_nr]; }
            template<typename ITERATOR>
                bool evaluate(const size_t bdd_nr, ITERATOR var_begin, ITERATOR var_end) const;
            template<typename ITERATOR>
                void rebase(const size_t bdd_nr, ITERATOR var_map_begin, ITERATOR var_map_end);
            template<typename VAR_MAP>
                void rebase(const size_t bdd_nr, const VAR_MAP& var_map);
            template<typename ITERATOR>
                void rebase(ITERATOR var_map_begin, ITERATOR var_map_end);
            template<typename VAR_MAP>
                void rebase(const VAR_MAP& var_map);
            // returns old variables
            std::vector<size_t> rebase_to_contiguous(const size_t bdd_nr);

            // reorder nodes such that variable indices are grouped together
            void reorder(const size_t bdd_nr);
            bool is_reordered(const size_t bdd_nr) const;

            bool variables_sorted(const size_t bdd_nr) const;
            std::vector<size_t> variables(const size_t bdd_nr) const;
            std::array<size_t,2> min_max_variables(const size_t bdd_nr) const;
            size_t root_variable(const size_t bdd_nr) const;
            // remove bdds with indices occurring in iterator
            template<typename ITERATOR>
                void remove(ITERATOR bdd_it_begin, ITERATOR bdd_it_end);
            void remove(const size_t bdd_nr);

            bdd_collection_entry operator[](const size_t bdd_nr);
            bdd_instruction operator()(const size_t bdd_nr, const size_t offset) const;
            const bdd_instruction& get_bdd_instruction(const size_t i) const;

            template<typename STREAM>
                void export_graphviz(const size_t bdd_nr, STREAM& s) const;

            auto get_bdd_instructions(const size_t bdd_nr) const { return std::make_pair(bdd_instructions.begin() + bdd_delimiters[bdd_nr], bdd_instructions.begin() + bdd_delimiters[bdd_nr+1]); }
            auto get_reverse_bdd_instructions(const size_t bdd_nr) const { return std::make_pair(bdd_instructions.begin() + bdd_delimiters[bdd_nr], bdd_instructions.begin() + bdd_delimiters[bdd_nr+1]); }

            // for constructing a new bdd
            size_t new_bdd();
            bdd_collection_node add_bdd_node(const size_t var);
            void close_bdd();

            // for exporting to solvers: need modified BDD with arcs connecting consecutive variables
            bool contiguous_vars(const size_t bdd_nr) const;
            size_t make_qbdd(const size_t bdd_nr);
            size_t make_qbdd(const size_t bdd_nr, bdd_collection& o);

            bool is_bdd(const size_t i) const;
            bool is_qbdd(const size_t bdd_nr) const;

            template<typename STREAM, typename COST_ITERATOR>
                void write_bdd_lp(STREAM& s, COST_ITERATOR cost_begin, COST_ITERATOR cost_end) const;

            // variables fixed to 0 or 1
            std::array<std::vector<size_t>,2> fixed_variables(const size_t bdd_nr) const;

            template<typename STREAM>
                void write_lp(STREAM& s);

            // utility functions
            size_t simplex_constraint(const size_t n);
            size_t not_all_false_constraint(const size_t n);
            size_t all_equal_constraint(const size_t n);

            // merge BDDs from another bdd_collection
            void append(const bdd_collection& o);

        private:
            size_t bdd_and_impl(const size_t i, const size_t j, bdd_collection& o);
            template<size_t N>
                size_t bdd_and(const std::array<size_t,N>& bdds);
            template<size_t N>
                size_t bdd_and(const std::array<size_t,N>& bdds, bdd_collection& o);
            template<size_t N>
            size_t bdd_and_impl(const std::array<size_t,N>& bdds, std::unordered_map<std::array<size_t,N>,size_t,array_hasher<N>>& generated_nodes, bdd_collection& o);

            size_t splitting_variable(const bdd_instruction& k, const bdd_instruction& l) const;
            size_t add_bdd_impl(node_ref bdd);

            bool bdd_basic_check(const size_t bdd_nr) const;
            bool has_no_isomorphic_subgraphs(const size_t bdd_nr) const;
            bool no_parallel_arcs(const size_t bdd_nr) const;
            // bring last DAG into BDD-form
            void remove_parallel_arcs();
            void reduce_isomorphic_subgraphs();
            void reduce();
            void remove_dead_nodes(const std::vector<char>& remove);
            std::vector<char> reachable_nodes(const size_t bdd_nr) const;

            std::vector<bdd_instruction> bdd_instructions;
            std::vector<size_t> bdd_delimiters = {0};

            // temporary memory for bdd synthesis
            std::vector<bdd_instruction> stack; // for computing bdd meld;

            std::unordered_map<std::array<size_t,2>,size_t, array_hasher<2>> generated_nodes; // given nodes of left and right bdd, has melded template be generated?
            std::unordered_map<bdd_instruction,size_t,bdd_instruction_hasher> reduction; // for generating a restricted graph. Given a variable index and left and right descendant, has node been generated?

            // node_ref -> index in bdd_instructions
            std::unordered_map<node_ref, size_t> node_ref_hash;
    };

    template<typename ITERATOR>
        bool bdd_collection::evaluate(const size_t bdd_nr, ITERATOR var_begin, ITERATOR var_end) const
        {
            assert(bdd_nr < nr_bdds());
            for(size_t i=bdd_delimiters[bdd_nr];;)
            {
                const bdd_instruction bdd = bdd_instructions[i];
                if(bdd.is_topsink())
                    return true;
                if(bdd.is_botsink())
                    return false;
                assert(bdd.index < std::distance(var_begin, var_end));
                const bool x = *(var_begin + bdd.index);
                if(x == true)
                    i = bdd.hi;
                else
                    i = bdd.lo;
            } 
        }

    template<typename ITERATOR>
        void bdd_collection::rebase(const size_t bdd_nr, ITERATOR var_map_begin, ITERATOR var_map_end)
        {
            auto unique_values = [](auto begin, auto end) -> bool {
                std::vector<typename std::iterator_traits<decltype(begin)>::value_type> v(begin, end);
                std::sort(v.begin(), v.end());
                return std::unique(v.begin(), v.end()) == v.end();
            };
            assert(unique_values(var_map_begin, var_map_end));

            assert(bdd_nr < nr_bdds());
            for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]; ++i)
            {
                bdd_instruction& bdd = bdd_instructions[i];
                assert(bdd.index < std::distance(var_map_begin, var_map_end) || bdd.is_terminal());
                const size_t rebase_index = [&]() {
                    if(bdd.is_terminal())
                        return bdd.index;
                    else
                        return *(var_map_begin + bdd.index);
                }();
                bdd.index = rebase_index;
            }
        }

    template<typename VAR_MAP>
        void bdd_collection::rebase(const size_t bdd_nr, const VAR_MAP& var_map)
        {
            assert(bdd_nr < nr_bdds());
            for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]; ++i)
            {
                bdd_instruction& bdd = bdd_instructions[i];
                const size_t rebase_index = [&]() {
                    if(bdd.is_terminal())
                        return bdd.index;
                    else
                    {
                        assert(var_map.count(bdd.index) > 0);
                        return var_map.find(bdd.index)->second;
                    }
                }();
                bdd.index = rebase_index;
            } 
        }

    template<typename ITERATOR>
        void bdd_collection::rebase(ITERATOR var_map_begin, ITERATOR var_map_end)
        {
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
                rebase(bdd_nr, var_map_begin, var_map_end);
        }

    template<typename VAR_MAP>
        void bdd_collection::rebase(const VAR_MAP& var_map)
        {
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
                rebase(bdd_nr, var_map);
        }

    template<typename ITERATOR>
        void bdd_collection::remove(ITERATOR bdd_it_begin, ITERATOR bdd_it_end)
        {
            const size_t nr_bdds_remove = std::distance(bdd_it_begin, bdd_it_end);
            assert(std::distance(bdd_it_begin, bdd_it_end) <= nr_bdds());
            assert(std::is_sorted(bdd_it_begin, bdd_it_end));
            for(size_t i=0; i+1<std::distance(bdd_it_begin, bdd_it_end); ++i)
            {
                assert(*(bdd_it_begin+i) < *(bdd_it_begin+i+1));
            }

            if(nr_bdds_remove == 0)
                return;

            assert(*(bdd_it_begin+nr_bdds_remove-1) < nr_bdds());

            const size_t first_bdd_to_remove = *bdd_it_begin;
            auto bdd_it = bdd_it_begin;
            size_t bdd_idx_to = bdd_delimiters[first_bdd_to_remove];
            size_t bdd_nr_counter = first_bdd_to_remove;
            size_t cur_bdd_delimiter = bdd_delimiters[first_bdd_to_remove];

            for(size_t bdd_nr=*bdd_it_begin; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                if(bdd_it != bdd_it_end && bdd_nr == *bdd_it)
                {
                    ++bdd_it;
                }
                else // move bdd
                {
                    for(size_t bdd_idx_from=bdd_delimiters[bdd_nr]; bdd_idx_from<bdd_delimiters[bdd_nr+1]; ++bdd_idx_from, ++bdd_idx_to)
                    {
                        bdd_instructions[bdd_idx_to] = bdd_instructions[bdd_idx_from];
                        if(!bdd_instructions[bdd_idx_to].is_terminal())
                        {
                            bdd_instructions[bdd_idx_to].lo -= bdd_idx_from - bdd_idx_to;
                            bdd_instructions[bdd_idx_to].hi -= bdd_idx_from - bdd_idx_to;
                        }
                    }
                    ++bdd_nr_counter;
                    bdd_delimiters[bdd_nr_counter] = bdd_idx_to;
                }
            }

            bdd_delimiters.resize(bdd_delimiters.size() - nr_bdds_remove);
            bdd_instructions.resize(bdd_delimiters.back());

            return;
            for(size_t i=0; i<nr_bdds(); ++i)
            {
                assert(is_bdd(i) || is_qbdd(i));
            }
        }

    /*
     // old remove with copying vectors around
    template<typename ITERATOR>
        void bdd_collection::remove(ITERATOR bdd_it_begin, ITERATOR bdd_it_end)
        {
            const size_t nr_bdds_remove = std::distance(bdd_it_begin, bdd_it_end);
            assert(std::distance(bdd_it_begin, bdd_it_end) <= nr_bdds());
            assert(std::is_sorted(bdd_it_begin, bdd_it_end));
            assert(std::unique(bdd_it_begin, bdd_it_end) == bdd_it_end);

            if(nr_bdds_remove == 0)
                return;

            assert(*(bdd_it_begin+nr_bdds_remove-1) < nr_bdds());

            std::vector<size_t> new_bdd_delimiters;
            new_bdd_delimiters.reserve(bdd_delimiters.size() - nr_bdds_remove);
            for(size_t i=0; i<=*bdd_it_begin; ++i)
                new_bdd_delimiters.push_back(bdd_delimiters[i]);

            std::vector<bdd_instruction> new_bdd_instructions;
            for(size_t i=0; i<new_bdd_delimiters.back(); ++i)
                new_bdd_instructions.push_back(bdd_instructions[i]);

            auto bdd_it = bdd_it_begin;
            for(size_t bdd_nr=*bdd_it_begin; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                if(bdd_it != bdd_it_end && bdd_nr == *bdd_it) // skip bdd
                {
                    bdd_it++;
                }
                else // move bdd
                {
                    assert(bdd_delimiters[bdd_nr] >= new_bdd_delimiters.back());
                    const size_t offset_delta = bdd_delimiters[bdd_nr] - new_bdd_delimiters.back();
                    for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]; ++i)
                    {
                        const bdd_instruction& bdd_instr = bdd_instructions[i];
                        if(!bdd_instr.is_terminal())
                        {
                            assert(bdd_instr.lo >= offset_delta);
                            assert(bdd_instr.hi >= offset_delta);
                            new_bdd_instructions.push_back({bdd_instr.lo - offset_delta, bdd_instr.hi - offset_delta, bdd_instr.index});
                        }
                        else
                            new_bdd_instructions.push_back(bdd_instr);
                    }
                    new_bdd_delimiters.push_back(new_bdd_instructions.size());
                    assert(new_bdd_delimiters.back() - new_bdd_delimiters[ new_bdd_delimiters.size()-2 ] > 2);
                }
            }
            assert(bdd_it == bdd_it_end);

            std::swap(bdd_delimiters, new_bdd_delimiters);
            std::swap(bdd_instructions, new_bdd_instructions);

            for(size_t i=0; i<nr_bdds(); ++i)
            {
                assert(is_bdd(i) || is_qbdd(i));
            }
        }
        */

    template<size_t N, typename ITERATOR>
        std::array<size_t,N> construct_array(ITERATOR begin, ITERATOR end)
        {
            assert(std::distance(begin,end) == N);
            std::array<size_t,N> a;
            auto it = begin;
            for(size_t i=0; i<N; ++i, ++it)
                a[i] = *it;
            return a;
        }

    template<typename BDD_ITERATOR>
        size_t bdd_collection::bdd_and(BDD_ITERATOR bdd_begin, BDD_ITERATOR bdd_end)
        {
            return bdd_and(bdd_begin, bdd_end, *this); 
        }

    template<typename BDD_ITERATOR>
        size_t bdd_collection::bdd_and(BDD_ITERATOR bdd_begin, BDD_ITERATOR bdd_end, bdd_collection& o)
        {
            const size_t nr_bdds = std::distance(bdd_begin, bdd_end);

            constexpr static size_t bdd_and_th = 49; // can be up to 49

            if(nr_bdds > bdd_and_th)
            {
                // TODO: remove intermediate intersection bdds
                std::vector<size_t> bdd_nrs(bdd_begin, bdd_end);
                std::sort(bdd_nrs.begin(), bdd_nrs.end(), [&](const size_t bdd_nr_1, const size_t bdd_nr_2) {
                        const auto [first_var_1, last_var_1] = this->min_max_variables(bdd_nr_1);
                        const auto [first_var_2, last_var_2] = this->min_max_variables(bdd_nr_2);
                        if(first_var_1 != first_var_2)
                        return first_var_1 < first_var_2;
                        if(last_var_1 != last_var_2)
                        return first_var_1 < first_var_2;
                        if(this->nr_bdd_nodes(bdd_nr_1) != this->nr_bdd_nodes(bdd_nr_2))
                        return this->nr_bdd_nodes(bdd_nr_1) < this->nr_bdd_nodes(bdd_nr_2);
                        return bdd_nr_1 < bdd_nr_2; 
                        });
                std::reverse(bdd_nrs.begin(), bdd_nrs.end());

                while(bdd_nrs.size() >= bdd_and_th)
                {
                    size_t new_bdd_nr = bdd_and(bdd_nrs.end()-bdd_and_th, bdd_nrs.end());
                    bdd_nrs.resize(bdd_nrs.size()-bdd_and_th);
                    bdd_nrs.push_back(new_bdd_nr);
                }

                return bdd_and(bdd_nrs.begin(), bdd_nrs.end());
            }

            switch(nr_bdds) {
                case 1: return *bdd_begin;
                case 2: { const size_t i = *bdd_begin; ++bdd_begin; const size_t j = *bdd_begin; return bdd_and(i, j, o); }
                case 3: { auto b = construct_array<3>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 4: { auto b = construct_array<4>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 5: { auto b = construct_array<5>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 6: { auto b = construct_array<6>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 7: { auto b = construct_array<7>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 8: { auto b = construct_array<8>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 9: { auto b = construct_array<9>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 10: { auto b = construct_array<10>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 11: { auto b = construct_array<11>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 12: { auto b = construct_array<12>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 13: { auto b = construct_array<13>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 14: { auto b = construct_array<14>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 15: { auto b = construct_array<15>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 16: { auto b = construct_array<16>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 17: { auto b = construct_array<17>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 18: { auto b = construct_array<18>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 19: { auto b = construct_array<19>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 20: { auto b = construct_array<20>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 21: { auto b = construct_array<21>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 22: { auto b = construct_array<22>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 23: { auto b = construct_array<23>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 24: { auto b = construct_array<24>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 25: { auto b = construct_array<25>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 26: { auto b = construct_array<26>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 27: { auto b = construct_array<27>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 28: { auto b = construct_array<28>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 29: { auto b = construct_array<29>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 30: { auto b = construct_array<30>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 31: { auto b = construct_array<31>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 32: { auto b = construct_array<32>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 33: { auto b = construct_array<33>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 34: { auto b = construct_array<34>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 35: { auto b = construct_array<35>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 36: { auto b = construct_array<36>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 37: { auto b = construct_array<37>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 38: { auto b = construct_array<38>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 39: { auto b = construct_array<39>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 40: { auto b = construct_array<40>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 41: { auto b = construct_array<41>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 42: { auto b = construct_array<42>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 43: { auto b = construct_array<43>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 44: { auto b = construct_array<44>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 45: { auto b = construct_array<45>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 46: { auto b = construct_array<46>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 47: { auto b = construct_array<47>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 48: { auto b = construct_array<48>(bdd_begin, bdd_end); return bdd_and(b, o); }
                case 49: { auto b = construct_array<49>(bdd_begin, bdd_end); return bdd_and(b, o); }
                default: throw std::runtime_error("generic and not implemented.");
            }
        }

    template<typename VAR_SET>
        size_t bdd_collection::bdd_or_var(const size_t i, const VAR_SET& positive_variables, const VAR_SET& negative_variables)
        {
            assert(i < nr_bdds());
            // copy old bdd instructions
            for(size_t idx=bdd_delimiters[i]; idx<bdd_delimiters[i+1]; ++idx)
            {
                bdd_instruction instr = bdd_instructions[idx];
                if(!instr.is_terminal())
                {
                    instr.lo += bdd_delimiters.back() - bdd_delimiters[i];
                    instr.hi += bdd_delimiters.back() - bdd_delimiters[i];
                }
                bdd_instructions.push_back(instr);
            }
            bdd_delimiters.push_back(bdd_instructions.size());

            assert(bdd_instructions.back().is_terminal());
            assert(bdd_instructions[bdd_instructions.size()-2].is_terminal());
            assert(bdd_instructions[bdd_instructions.size()-1] != bdd_instructions[bdd_instructions.size()-2]);

            const size_t new_bdd_nr = bdd_delimiters.size()-2;

            const size_t botsink_idx = botsink_index(new_bdd_nr);
            const size_t topsink_idx = topsink_index(new_bdd_nr);

            // reroute arcs to topsink for instructions that cover positive or negative variables
            for(size_t idx=bdd_delimiters[new_bdd_nr]; idx<bdd_delimiters[new_bdd_nr+1]-2; ++idx)
            {
                auto& instr = bdd_instructions[idx];
                assert(!instr.is_terminal());
                assert(!(positive_variables.count(instr.index) > 0 && negative_variables.count(instr.index) > 0));
                if(positive_variables.count(instr.index) > 0)
                    instr.hi = topsink_idx; 
                if(negative_variables.count(instr.index) > 0)
                {
                    // TODO: what about botsink_idx
                    throw std::runtime_error("possible bug here");
                    instr.lo = topsink_idx; 
                }
            }

            reduce(); 
            assert(is_bdd(new_bdd_nr));
            return new_bdd_nr;
        }


    template<typename ITERATOR>
        void bdd_collection_entry::rebase(ITERATOR var_map_begin, ITERATOR var_map_end) 
        { 
            bdd_col.rebase(bdd_nr, var_map_begin, var_map_end);
        }

    template<typename VAR_MAP>
        void bdd_collection_entry::rebase(const VAR_MAP& var_map) 
        { 
            bdd_col.rebase(bdd_nr, var_map);
        }


    template<typename STREAM>
        void bdd_collection::export_graphviz(const size_t bdd_nr, STREAM& s) const
        {
            assert(bdd_nr < nr_bdds());
            s << "digraph BDD\n";
            s << "{\n";

            std::unordered_map<size_t,std::string> clusters;
            std::unordered_map<size_t, size_t> cluster_nodes;

            for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]; ++i)
            {
                const bdd_instruction& bdd_instr = bdd_instructions[i];
                if(bdd_instr.is_terminal())
                {
                    std::string& str = clusters[std::numeric_limits<size_t>::max()];
                    if(bdd_instr.is_topsink())
                        str += std::to_string(i - bdd_delimiters[bdd_nr]) + " [label=\"top\"];\n";
                    else if(bdd_instr.is_botsink())
                        str += std::to_string(i - bdd_delimiters[bdd_nr]) + " [label=\"bot\"];\n";
                    else
                        assert(false); 
                }
                else
                {
                    std::string& str = clusters[bdd_instr.index];
                    str += std::to_string(i - bdd_delimiters[bdd_nr]) + " [label=\"" += std::to_string(bdd_instr.index) + "\"];\n"; 
                    cluster_nodes[bdd_instr.index] = i - bdd_delimiters[bdd_nr];
                }

                /*
                if(bdd_instr.is_terminal())
                    s << i - bdd_delimiters[bdd_nr] << " [label=\"top\"];\n";
                else if(bdd_instr.is_botsink())
                    s << i - bdd_delimiters[bdd_nr] << " [label=\"bot\"];\n";
                else
                    s << i - bdd_delimiters[bdd_nr] << " [label=\"" << bdd_instr.index << "\"];\n";
                    */
            }

            for(auto& [idx, str] : clusters)
            {
                s << "subgraph cluster_" << idx << " {\n";
                s << str;
                s << "color = blue\n";
                s << "}\n"; 
            }

            // add invisible arrows between clusters
            std::vector<std::array<size_t,2>> cluster_nodes2;
            for(auto [x,y] : cluster_nodes)
                cluster_nodes2.push_back({x,y});
            std::sort(cluster_nodes2.begin(), cluster_nodes2.end(), [](const auto a, const auto b) { return a[0] < b[0]; });
            for(size_t c=0; c<cluster_nodes2.size()-1; ++c)
                s << cluster_nodes2[c][1] << " -> " << cluster_nodes2[c+1][1] << " [style=invis];\n";

            for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]; ++i)
            {
                const bdd_instruction& bdd_instr = bdd_instructions[i];
                if(bdd_instr.is_terminal())
                    continue;

                s << i - bdd_delimiters[bdd_nr] << " -> " << bdd_instr.hi - bdd_delimiters[bdd_nr] << ";\n";
                s << i - bdd_delimiters[bdd_nr] << " -> " << bdd_instr.lo - bdd_delimiters[bdd_nr] << "[style=\"dashed\"];\n";
            } 
            s << "}\n";
        }

    template<typename STREAM, typename COST_ITERATOR>
        void bdd_collection::write_bdd_lp(STREAM& s, COST_ITERATOR cost_begin, COST_ITERATOR cost_end) const
        {
            auto arc_identifier = [&](const size_t bdd_nr, const size_t bdd_idx, const size_t value)
            {
                assert(bdd_nr < nr_bdds());
                assert(bdd_idx >= bdd_delimiters[bdd_nr] && bdd_idx < bdd_delimiters[bdd_nr+1]-2);
                assert(value <= 1);
                return std::string("arc_") + std::to_string(bdd_nr) + "_" + std::to_string(bdd_idx-bdd_delimiters[bdd_nr]) + "_" + std::to_string(value); 
            };

            auto var_identifier = [&](const size_t var)
            {
                return std::string("x_") + std::to_string(var);
            };

            s << "Minimize\n";
            for(size_t i=0; i<std::distance(cost_begin, cost_end); ++i)
            {
                const double val = *(cost_begin + i);
                s << (val < 0 ? "-" : "+") << std::abs(val) << " " << var_identifier(i) << "\n";
            }

            s << "Subject To\n";

            // constraints for individual BDDs
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                // exactly one path starts at root node
                const auto root_instr = bdd_instructions[bdd_delimiters[bdd_nr]];
                s << "R_" << bdd_nr << ": ";
                if(!bdd_instructions[root_instr.lo].is_botsink())
                    s << arc_identifier(bdd_nr, bdd_delimiters[bdd_nr], 0);
                if(!bdd_instructions[root_instr.hi].is_botsink())
                    s << " + " << arc_identifier(bdd_nr, bdd_delimiters[bdd_nr], 1);
                s << " = 1\n";

                // node, value
                std::vector<std::vector<std::array<size_t,2>>> incoming_arcs(nr_bdd_nodes(bdd_nr));
                if(!bdd_instructions[root_instr.lo].is_terminal())
                    incoming_arcs[root_instr.lo - bdd_delimiters[bdd_nr]].push_back({bdd_delimiters[bdd_nr],0});
                if(!bdd_instructions[root_instr.hi].is_terminal())
                    incoming_arcs[root_instr.hi - bdd_delimiters[bdd_nr]].push_back({bdd_delimiters[bdd_nr],1});
                for(size_t i=bdd_delimiters[bdd_nr]+1; i<bdd_delimiters[bdd_nr+1]-2; ++i)
                {
                    const auto& instr = bdd_instructions[i];
                    s << "FC_" << bdd_nr << "_" << i-bdd_delimiters[bdd_nr] << ": ";
                    // flow conservation constraint for intermediate nodes
                    // outgoing
                    if(!bdd_instructions[instr.lo].is_botsink())
                        s << arc_identifier(bdd_nr, i, 0);
                    if(!bdd_instructions[instr.hi].is_botsink())
                        s << " + " << arc_identifier(bdd_nr, i, 1);
                    // incoming
                    assert(incoming_arcs[i - bdd_delimiters[bdd_nr]].size() > 0);
                    for(const auto [node, value] : incoming_arcs[i-bdd_delimiters[bdd_nr]])
                        s << " - " << arc_identifier(bdd_nr, node, value);
                    s << " = 0\n";
                    if(!bdd_instructions[instr.lo].is_terminal())
                        incoming_arcs[instr.lo - bdd_delimiters[bdd_nr]].push_back({i,0});
                    if(!bdd_instructions[instr.hi].is_terminal())
                        incoming_arcs[instr.hi - bdd_delimiters[bdd_nr]].push_back({i,1}); 
                }
            }

            // constraints linking variables across BDDs
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                assert(is_reordered(bdd_nr));
                size_t cur_var = min_max_variables(bdd_nr)[0];
                for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
                {
                    const bdd_instruction& instr = bdd_instructions[i];
                    std::cout << "bdd_nr " << bdd_nr << ", i = " << i - bdd_delimiters[bdd_nr] << ", var = " << instr.index << "\n";
                    assert(!instr.is_terminal());
                    if(instr.index != cur_var)
                    {
                        s << " - " << var_identifier(cur_var) << " = 0\n";
                        cur_var = instr.index; 
                    }
                    if(!bdd_instructions[instr.hi].is_botsink())
                        s << " + " << arc_identifier(bdd_nr, i, 1); 
                }
                s << " - " << var_identifier(cur_var) << " = 0\n";
                assert(cur_var == min_max_variables(bdd_nr)[1]);
            }

            s << "Bounds\n";
            s << "Binaries\n";
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t i=bdd_delimiters[bdd_nr]; i<bdd_delimiters[bdd_nr+1]-2; ++i)
                {
                    const bdd_instruction& instr = bdd_instructions[i];
                    if(!bdd_instructions[instr.lo].is_botsink())
                        s << arc_identifier(bdd_nr, i, 0) << "\n";
                    if(!bdd_instructions[instr.hi].is_botsink())
                        s << arc_identifier(bdd_nr, i, 1) << "\n";
                }
            }
            s << "End\n"; 
        }

}

// inject hash function for bdd_collection_node into std namespace
namespace std
{
    template<> struct hash<BDD::bdd_collection_node>
    {
        size_t operator()(const BDD::bdd_collection_node& bdd) const
        {
            return std::hash<size_t>()(bdd.i);
        }
    };
}
