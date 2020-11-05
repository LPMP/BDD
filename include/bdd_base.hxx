#pragma once

#include "bdd_storage.h"
#include "bdd_branch_node.h"
#include <cassert>
#include <vector>
#include <tsl/robin_map.h>
//#include <unordered_map> // TODO: use tsl::robin-map
//#include <unordered_set> // TODO: use tsl::robin-map

namespace LPMP {

    // base class for min marginal averaging and rounding. Take bdd storage as input and lay out bdd nodes per variable.

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    class bdd_base {
        public:
            bdd_base();
            bdd_base(bdd_storage& stor);
            bdd_base(const bdd_base&) = delete; // no copy constructor because of pointers in bdd_branch_node

            template<typename BDD_VARIABLES_ITERATOR>
                void add_bdd(BDD::node_ref bdd, BDD_VARIABLES_ITERATOR bdd_vars_begin, BDD_VARIABLES_ITERATOR bdd_vars_end, BDD::bdd_mgr& bdd_mgr);

            void init(bdd_storage& bdd_storage_);

            struct bdd_endpoints_ {
                size_t first_variable;
                size_t first_bdd_index;
                size_t last_variable;
                size_t last_bdd_index;
            };
            std::vector<bdd_endpoints_> bdd_endpoints(bdd_storage& bdd_storage_) const;

            std::size_t nr_variables() const { return bdd_variables_.size(); }
            std::size_t nr_bdds() const { return nr_bdds_; }
            std::size_t nr_bdds(const std::size_t var) const { assert(var<nr_variables()); return bdd_variables_.size(var); }

            size_t nr_feasible_outgoing_arcs(const size_t var, const size_t bdd_index) const;

            void forward_step(const std::size_t var, const std::size_t bdd_index);
            void forward_step_tmp(const std::size_t var, const std::size_t bdd_index);
            void backward_step(const std::size_t var, const std::size_t bdd_index);
            void forward_step(const std::size_t var);
            void backward_step(const std::size_t var);

            void backward_run(); // also used to initialize
            void forward_run();

            BDD_VARIABLE &get_bdd_variable(const std::size_t var, const std::size_t bdd_index);
            const BDD_VARIABLE &get_bdd_variable(const std::size_t var, const std::size_t bdd_index) const;

            const BDD_BRANCH_NODE &get_bdd_branch_node(const std::size_t var, const std::size_t bdd_index, const std::size_t bdd_node_index) const;

            bool first_variable_of_bdd(const std::size_t var, const std::size_t bdd_index) const;
            bool last_variable_of_bdd(const std::size_t var, const std::size_t bdd_index) const;

        protected:
            size_t bdd_branch_node_index(const BDD_BRANCH_NODE* bdd) const;
            size_t bdd_branch_node_index(const BDD_BRANCH_NODE& bdd) const { return bdd_branch_node_index(&bdd); }
            size_t variable_index(const BDD_BRANCH_NODE& bdd) const;
            size_t variable_index(const BDD_VARIABLE& bdd_var) const;

            size_t first_variable_of_bdd(const BDD_VARIABLE& bdd_var) const;
            size_t last_variable_of_bdd(const BDD_VARIABLE& bdd_var) const;

            std::vector<BDD_BRANCH_NODE> bdd_branch_nodes_;
            two_dim_variable_array<BDD_VARIABLE> bdd_variables_;
            size_t nr_bdds_ = 0;
    };

    ////////////////////
    // implementation //
    ////////////////////

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_base()
    {}

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_base(bdd_storage& stor)
    {
        init(stor);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::init(bdd_storage& bdd_storage_)
    {
        assert(bdd_storage_.nr_variables() > 0 && bdd_storage_.nr_bdds() > 0);
        // allocate datastructures holding bdd instructions
        
        // helper vectors used throughout initialization
        std::vector<std::size_t> nr_bdd_nodes_per_variable(bdd_storage_.nr_variables(), 0);
        std::vector<std::size_t> nr_bdds_per_variable(bdd_storage_.nr_variables(), 0);

        // TODO: std::unordered_set might be faster (or tsl::unordered_set)
        std::vector<char> variable_counted(bdd_storage_.nr_variables(), 0);
        std::vector<std::size_t> variables_covered;

        auto variable_covered = [&](const std::size_t v) {
            assert(v < bdd_storage_.nr_variables());
            assert(variable_counted[v] == 0 || variable_counted[v] == 1);
            return variable_counted[v] == 1;
        };
        auto cover_variable = [&](const std::size_t v) {
            assert(!variable_covered(v));
            variable_counted[v] = 1;
            variables_covered.push_back(v);
        };
        auto uncover_variables = [&]() {
            for(const std::size_t v : variables_covered) {
                assert(variable_covered(v));
                variable_counted[v] = 0;
            }
            variables_covered.clear(); 
        };

        for(const auto& bdd_node : bdd_storage_.bdd_nodes()) {
            ++nr_bdd_nodes_per_variable[bdd_node.variable];
            // if we have additional gap nodes, count them too
            const auto& next_high = bdd_storage_.bdd_nodes()[bdd_node.high];
            //std::cout << bdd_node.variable << " ";
        }
        //std::cout << "\n";
        assert(bdd_storage_.bdd_nodes().size() == std::accumulate(nr_bdd_nodes_per_variable.begin(), nr_bdd_nodes_per_variable.end(), 0));

        //std::cout << "cover variables\n";
        for(std::size_t bdd_index=0; bdd_index<bdd_storage_.bdd_delimiters().size()-1; ++bdd_index) {
            for(std::size_t i=bdd_storage_.bdd_delimiters()[bdd_index]; i<bdd_storage_.bdd_delimiters()[bdd_index+1]; ++i) {
                const std::size_t bdd_variable = bdd_storage_.bdd_nodes()[i].variable;
                if(!variable_covered(bdd_variable)) {
                    cover_variable(bdd_variable);
                    ++nr_bdds_per_variable[bdd_variable];
                }
            }
            //std::cout << "bdd index = " << bdd_index << "\n";
            uncover_variables();
        }
        //std::cout << "uncover variables\n";
        
        std::vector<std::size_t> bdd_offset_per_variable;
        bdd_offset_per_variable.reserve(bdd_storage_.nr_variables());
        bdd_offset_per_variable.push_back(0);
        std::partial_sum(nr_bdd_nodes_per_variable.begin(), nr_bdd_nodes_per_variable.end()-1, std::back_inserter(bdd_offset_per_variable));
        assert(bdd_offset_per_variable.size() == bdd_storage_.nr_variables());
        assert(bdd_offset_per_variable.back() + nr_bdd_nodes_per_variable.back() == bdd_storage_.bdd_nodes().size());

        bdd_branch_nodes_.resize(bdd_storage_.bdd_nodes().size());
        bdd_variables_.resize(nr_bdds_per_variable.begin(), nr_bdds_per_variable.end());

        for(std::size_t v=0; v<nr_bdds_per_variable.size(); ++v) {
            //std::cout << "v = " << v << ", nr bdd nodes per var = " << nr_bdd_nodes_per_variable[v] << ", offset = " << bdd_offset_per_variable[v] << "\n";
        }

        // fill branch instructions into datastructures and set pointers
        std::fill(nr_bdd_nodes_per_variable.begin(), nr_bdd_nodes_per_variable.end(), 0); // counter for bdd instruction per variable
        std::fill(nr_bdds_per_variable.begin(), nr_bdds_per_variable.end(), 0); // counter for bdd per variable

        //std::unordered_map<std::size_t, BDD_BRANCH_NODE*> stored_bdd_node_index_to_bdd_address;
        tsl::robin_map<std::size_t, BDD_BRANCH_NODE*> stored_bdd_node_index_to_bdd_address;
        //stored_bdd_node_index_to_bdd_address.insert(std::pair<std::size_t, BDD_BRANCH_NODE*>(bdd_storage::bdd_node::terminal_0, bdd_branch_node_terminal_0));
        //stored_bdd_node_index_to_bdd_address.insert(std::pair<std::size_t, BDD_BRANCH_NODE*>(bdd_storage::bdd_node::terminal_1, bdd_branch_node_terminal_1));


        std::size_t c = 0; // check if everything is read contiguously
        for(std::size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index) {
            ++nr_bdds_;
            uncover_variables(); 
            stored_bdd_node_index_to_bdd_address.clear();
            stored_bdd_node_index_to_bdd_address.insert(std::pair<std::size_t, BDD_BRANCH_NODE*>(bdd_storage::bdd_node::terminal_0, BDD_BRANCH_NODE::terminal_0()));
            stored_bdd_node_index_to_bdd_address.insert(std::pair<std::size_t, BDD_BRANCH_NODE*>(bdd_storage::bdd_node::terminal_1, BDD_BRANCH_NODE::terminal_1()));
            BDD_VARIABLE* next_bdd_var = nullptr;

            //std::cout << "bdd index = " << bdd_index << "\n";
            const std::size_t first_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index]; 
            const std::size_t last_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index+1];
            //std::cout << "bdd delimiter = " << bdd_storage_.bdd_delimiters()[bdd_index+1] << "\n";
            for(std::size_t stored_bdd_node_index=first_stored_bdd_node; stored_bdd_node_index<last_stored_bdd_node; ++stored_bdd_node_index, ++c) {
                assert(c == stored_bdd_node_index);
                //std::cout << "stored bdd node index = " << stored_bdd_node_index << "\n";
                const auto& stored_bdd = bdd_storage_.bdd_nodes()[stored_bdd_node_index];
                const std::size_t v = stored_bdd.variable;
                //std::cout << "bdd variable = " << v << ", bdd variable offset = " << bdd_offset_per_variable[v] << "\n";

                auto& bdd_var = bdd_variables_(v, nr_bdds_per_variable[v]);
                if(!variable_covered(v)) {
                    // assert(bdd_var.is_initial_state());
                    cover_variable(v);

                    bdd_var.first_node_index = bdd_offset_per_variable[v];
                    bdd_var.last_node_index = bdd_offset_per_variable[v];
                    //std::cout << "bdd level offset for var = " << v << ", bdd index = " << bdd_index << " = " << stored_bdd_node_index << "\n";
                    // TODO: remove, not needed anymore
                    if(next_bdd_var != nullptr) {
                        //assert(next_bdd_var > &bdd_var);
                        bdd_var.next = next_bdd_var;
                        next_bdd_var->prev = &bdd_var;
                    }

                    next_bdd_var = &bdd_var;
                } else {
                    // assert(!bdd_var.is_initial_state());
                }

                const std::size_t bdd_branch_nodes_index = bdd_var.last_node_index; 
                bdd_var.last_node_index++;

                BDD_BRANCH_NODE& bdd = bdd_branch_nodes_[bdd_branch_nodes_index];
                // assert(bdd.is_initial_state());
                //std::cout << "address = " << &bdd << "\n";

                stored_bdd_node_index_to_bdd_address.insert({stored_bdd_node_index, &bdd});

                assert(stored_bdd_node_index_to_bdd_address.count(stored_bdd.low) > 0);
                BDD_BRANCH_NODE& bdd_low = *(stored_bdd_node_index_to_bdd_address.find(stored_bdd.low)->second);
                bdd.low_outgoing = &bdd_low;
                assert(bdd.low_outgoing != nullptr);
                if constexpr(has_incoming_pointers<BDD_BRANCH_NODE>::value)
                    if(!BDD_BRANCH_NODE::is_terminal(&bdd_low)) {
                        bdd.next_low_incoming = bdd_low.first_low_incoming;
                        bdd_low.first_low_incoming = &bdd;
                    }

                assert(stored_bdd_node_index_to_bdd_address.count(stored_bdd.high) > 0);
                BDD_BRANCH_NODE& bdd_high = *(stored_bdd_node_index_to_bdd_address.find(stored_bdd.high)->second);
                bdd.high_outgoing = &bdd_high;
                if constexpr(has_incoming_pointers<BDD_BRANCH_NODE>::value)
                    if(!BDD_BRANCH_NODE::is_terminal(&bdd_high)) {
                        bdd.next_high_incoming = bdd_high.first_high_incoming;
                        bdd_high.first_high_incoming = &bdd;
                    }

                //if(!bdd.low_outgoing->is_terminal()) { assert(variable_index(bdd) < variable_index(*bdd.low_outgoing)); }
                //if(!bdd.high_outgoing->is_terminal()) { assert(variable_index(bdd) < variable_index(*bdd.high_outgoing)); }
                //check_bdd_branch_node(bdd, v+1 == nr_variables(), v == 0); // TODO: cannot be enabled because not all pointers are set yet.
                //check_bdd_branch_instruction_level(bdd_level, v+1 == nr_variables(), v == 0);
                ++nr_bdd_nodes_per_variable[v]; 
                ++bdd_offset_per_variable[v];
            }

            for(const std::size_t v : variables_covered) {
                ++nr_bdds_per_variable[v];
            }
        }

        for(std::size_t v=0; v<nr_bdds_per_variable.size(); ++v) {
            assert(nr_bdds_per_variable[v] == bdd_variables_[v].size());
        }

        for(const auto& bdd : bdd_branch_nodes_) {
            check_bdd_branch_node(bdd);
        }

        for(std::size_t v=0; v<nr_variables(); ++v) {
            for(std::size_t bdd_index=0; bdd_index<nr_bdds(v); ++bdd_index) {
                // TODO: reactive check_bdd_branch_variable
                //check_bdd_branch_instruction_level(bdd_branch_instruction_levels(v,bdd_index));
            }
        }
    }

    // return type: variable, bdd index
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    std::vector<typename bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_endpoints_> bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_endpoints(bdd_storage& bdd_storage_) const
    {
        // argument bdd storage must be the same as the one used for constructing the base
        assert(bdd_storage_.nr_variables() == this->nr_variables() && bdd_storage_.nr_bdds() == this->nr_bdds());
        // iterate through bdd storage bdds and get first and last variable of bdd. Then record bdd indices at corresponding variables

        std::vector<bdd_endpoints_> endpoints;
        endpoints.reserve(bdd_storage_.nr_bdds());
        std::vector<size_t> bdd_index_counter(bdd_storage_.nr_variables(), 0);

        //std::unordered_set<size_t> bdd_variables;
        tsl::robin_set<size_t> bdd_variables;
        for(size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index)
        {
            size_t first_bdd_var = std::numeric_limits<size_t>::max();
            size_t last_bdd_var = 0;
            bdd_variables.clear();
            for(size_t i=bdd_storage_.bdd_delimiters()[bdd_index]; i<bdd_storage_.bdd_delimiters()[bdd_index+1]; ++i)
            {
                const size_t bdd_var = bdd_storage_.bdd_nodes()[i].variable;
                if(bdd_var < bdd_storage_.nr_variables()) // otherwise top and bottom sink
                {
                    first_bdd_var = std::min(bdd_var, first_bdd_var);
                    last_bdd_var = std::max(bdd_var, last_bdd_var); 
                    bdd_variables.insert(bdd_var);
                }
            }

            assert(first_bdd_var <= last_bdd_var);
            assert(last_bdd_var < this->nr_variables());
            assert(first_variable_of_bdd(first_bdd_var, bdd_index_counter[first_bdd_var]));
            assert(last_variable_of_bdd(last_bdd_var, bdd_index_counter[last_bdd_var]));
            assert(bdd_index_counter[first_bdd_var] < this->nr_bdds(first_bdd_var));
            assert(bdd_index_counter[last_bdd_var] < this->nr_bdds(last_bdd_var));
            assert(bdd_variables.count(first_bdd_var) > 0);
            assert(bdd_variables.count(last_bdd_var) > 0);

            endpoints.push_back({first_bdd_var, bdd_index_counter[first_bdd_var], last_bdd_var, bdd_index_counter[last_bdd_var]});
            for(size_t v : bdd_variables)
                ++bdd_index_counter[v];
        }

        return endpoints;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    std::size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_branch_node_index(const BDD_BRANCH_NODE* bdd) const
    {
        assert(bdd >= &bdd_branch_nodes_[0]);
        const std::size_t i = bdd - &bdd_branch_nodes_[0];
        assert(i < bdd_branch_nodes_.size());
        return i; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    std::size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::variable_index(const BDD_BRANCH_NODE& bdd) const
    {
        const std::size_t i = bdd_branch_node_index(&bdd);

        std::size_t lb = 0;
        std::size_t ub = nr_variables()-1;
        std::size_t v = nr_variables()/2;
        while (! (i >= bdd_variables_(v, 0).first_node_index && i < bdd_variables_[v].back().last_node_index))
        {
            if (i > bdd_variables_(v, 0).first_node_index)
                lb = v+1;
            else
                ub = v-1;
            v = (lb+ub)/2;
        }
        return v;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    std::size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::variable_index(const BDD_VARIABLE& bdd_var) const
    {
        assert(&bdd_var >= &bdd_variables_(0,0));
        assert(&bdd_var <= &bdd_variables_.back().back());
        std::size_t lb = 0;
        std::size_t ub = nr_variables()-1;
        std::size_t v = nr_variables()/2;
        while(! (&bdd_var >= &bdd_variables_(v,0) && &bdd_var <= &bdd_variables_[v].back()) ) {
            if( &bdd_var > &bdd_variables_(v,0) ) {
                lb = v+1;
            } else {
                ub = v-1;
            }
            v = (ub+lb)/2; 
        }
        assert(&bdd_var >= &bdd_variables_(v,0) && &bdd_var <= &bdd_variables_[v].back());
        return v; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    std::size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::first_variable_of_bdd(const BDD_VARIABLE& bdd_var) const
    {
        BDD_VARIABLE const* p = &bdd_var;
        while(p->prev != nullptr)
            p = p->prev;
        return variable_index(*p);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    std::size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::last_variable_of_bdd(const BDD_VARIABLE& bdd_var) const
    {
        BDD_VARIABLE const* p = &bdd_var;
        while(p->next != nullptr)
            p = p->next;
        return variable_index(*p); 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    bool bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::last_variable_of_bdd(const std::size_t var, const std::size_t bdd_index) const
    {
        const BDD_VARIABLE& bdd_var = bdd_variables_(var, bdd_index);
        return bdd_var.next == nullptr; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    bool bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::first_variable_of_bdd(const std::size_t var, const std::size_t bdd_index) const
    {
        const BDD_VARIABLE& bdd_var = bdd_variables_(var, bdd_index);
        return bdd_var.prev == nullptr; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::nr_feasible_outgoing_arcs(const size_t var, const size_t bdd_index) const
    {
        assert(var < bdd_variables_.size());
        assert(bdd_index < bdd_variables_[var].size());

        size_t n = 0;
        const auto& bdd_var = get_bdd_variable(var, bdd_index);
        const std::size_t first_node_index = bdd_var.first_node_index;
        const std::size_t last_node_index = bdd_var.last_node_index;
        for(std::size_t i=first_node_index; i<last_node_index; ++i) {
            const auto& node = bdd_branch_nodes_[i];
            if(node.low_outgoing != node.terminal_0())
                ++n;
            if(node.high_outgoing != node.terminal_0())
                ++n;
        }

        return n;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_step(const std::size_t var, const std::size_t bdd_index)
    {
        assert(var < bdd_variables_.size());
        assert(bdd_index < bdd_variables_[var].size());

        auto& bdd_var = bdd_variables_(var,bdd_index);
        assert(var != 0 || bdd_var.prev == nullptr);

        // iterate over all bdd nodes and make forward step
        const std::size_t first_node_index = bdd_var.first_node_index;
        const std::size_t last_node_index = bdd_var.last_node_index;
        for(std::size_t i=first_node_index; i<last_node_index; ++i) {
            //std::cout << "forward step for var = " << var << ", bdd_index = " << bdd_index << ", bdd_node_index = " << i << "\n";
            bdd_branch_nodes_[i].forward_step();
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_step_tmp(const size_t var, const size_t bdd_index)
    {
        assert(var < this->nr_variables());
        assert(bdd_index < this->nr_bdds(var));

        const auto& bdd_var = get_bdd_variable(var, bdd_index);

        if(bdd_var.prev == nullptr)
        {
                const size_t first_node_index = bdd_var.first_node_index;
                const size_t last_node_index = bdd_var.last_node_index;
                for (size_t i=first_node_index; i<last_node_index; ++i)
                    this->bdd_branch_nodes_[i].m = 0.0;
        }

        if(bdd_var.next != nullptr)
        {
            // set m-values of bdds from next variable to zero
            {
                const auto& next_bdd_var = *bdd_var.next;
                const size_t first_node_index = next_bdd_var.first_node_index;
                const size_t last_node_index = next_bdd_var.last_node_index;
                for (size_t i=first_node_index; i<last_node_index; ++i)
                    this->bdd_branch_nodes_[i].m = std::numeric_limits<double>::infinity();
            }

            // go over each bdd branch node from this variable and update outgoing m-values
            {
                const size_t first_node_index = bdd_var.first_node_index;
                const size_t last_node_index = bdd_var.last_node_index;
                for(size_t i=first_node_index; i<last_node_index; ++i)
                    this->bdd_branch_nodes_[i].forward_step();
            }
        } 
    }


    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_step(const std::size_t var)
    {
        assert(var < bdd_variables_.size());
        for(std::size_t bdd_index = 0; bdd_index<bdd_variables_[var].size(); ++bdd_index)
            forward_step(var, bdd_index);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::backward_step(const std::size_t var, const std::size_t bdd_index)
    {
        assert(var < bdd_variables_.size());
        assert(bdd_index < bdd_variables_[var].size());

        auto& bdd_var = bdd_variables_(var,bdd_index);
        assert(var+1 != nr_variables() || bdd_var.next == nullptr);
        // iterate over all bdd nodes and make forward step
        const std::ptrdiff_t first_node_index = bdd_var.first_node_index;
        const std::ptrdiff_t last_node_index = bdd_var.last_node_index;
        //std::cout << "no bdd nodes for var " << var << " = " << last_node_index - first_node_index << "\n";
        //std::cout << "bdd branch instruction offset = " << first_node_index << "\n";
        for(std::ptrdiff_t i=last_node_index-1; i>=first_node_index; --i) {
            check_bdd_branch_node(bdd_branch_nodes_[i], var+1 == nr_variables(), var == 0);
            bdd_branch_nodes_[i].backward_step();
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::backward_step(const std::size_t var)
    {
        assert(var < bdd_variables_.size());
        for(std::size_t bdd_index = 0; bdd_index<bdd_variables_[var].size(); ++bdd_index)
            backward_step(var, bdd_index);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::backward_run()
    {
        for(std::ptrdiff_t var=nr_variables()-1; var>=0; --var)
            backward_step(var); 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_run()
    {
        for(std::size_t var=0; var<nr_variables(); ++var)
            forward_step(var); 
    }

    template <typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    BDD_VARIABLE &bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::get_bdd_variable(const std::size_t var, const std::size_t bdd_index)
    {
        assert(var < nr_variables());
        assert(bdd_index < bdd_variables_[var].size());
        return bdd_variables_(var, bdd_index);
    }

    template <typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    const BDD_VARIABLE &bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::get_bdd_variable(const std::size_t var, const std::size_t bdd_index) const
    {
        assert(var < nr_variables());
        assert(bdd_index < bdd_variables_[var].size());
        return bdd_variables_(var, bdd_index);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    const BDD_BRANCH_NODE& bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::get_bdd_branch_node(const std::size_t var, const std::size_t bdd_index, const std::size_t bdd_node_index) const
    {
        assert(var < nr_variables());
        assert(bdd_index < bdd_variables_[var].size());
        assert(bdd_node_index < bdd_variables_(var,bdd_index).nr_bdd_nodes());
        return bdd_branch_nodes_[ bdd_variables_(var,bdd_index).first_node_index + bdd_node_index ];
    }
}
