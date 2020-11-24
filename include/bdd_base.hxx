#pragma once

#include "bdd_storage.h"
#include "bdd_branch_node.h"
#include <cassert>
#include <vector>
#include <tsl/robin_map.h>
#include <iostream> // TODO: delete
#include "time_measure_util.h"
//#include <unordered_map> // TODO: use tsl::robin-map
//#include <unordered_set> // TODO: use tsl::robin-map


#pragma omp declare reduction(vec_size_t_plus : std::vector<size_t> : \
        std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<size_t>())) \
        initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))


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

            size_t nr_variables() const { return bdd_variables_.size(); }
            size_t nr_bdds() const { return nr_bdds_; }
            size_t nr_bdds(const size_t var) const { assert(var<nr_variables()); return bdd_variables_.size(var); }

            template <typename ITERATOR>
                bool check_feasibility(ITERATOR var_begin, ITERATOR var_end) const;

            size_t nr_feasible_outgoing_arcs(const size_t var, const size_t bdd_index) const;

            // TODO: possibly put into bdd_opt_base //
            void backward_step(const size_t var, const size_t bdd_index);
            void backward_step(const BDD_VARIABLE& bdd_var);
            void backward_step(const size_t var);

            void forward_step(const size_t var, const size_t bdd_index);
            void forward_step(const BDD_VARIABLE& bdd_var);
            void forward_step(const size_t var);

            void backward_run(); // also used to initialize
            void forward_run();
            ////////////////////////////////////////////

            BDD_VARIABLE &get_bdd_variable(const size_t var, const size_t bdd_index);
            const BDD_VARIABLE &get_bdd_variable(const size_t var, const size_t bdd_index) const;

            const BDD_BRANCH_NODE &get_bdd_branch_node(const size_t var, const size_t bdd_index, const size_t bdd_node_index) const;

            bool first_variable_of_bdd(const size_t var, const size_t bdd_index) const;
            bool last_variable_of_bdd(const size_t var, const size_t bdd_index) const;

            // TODO: use downstream
            // void nr_bdds(const size_t var) const { return nr_bdds_per_variable_[var]; }
            std::array<BDD_BRANCH_NODE*,2> bdd_range(const size_t var) const;
            std::array<BDD_BRANCH_NODE*,2> bdd_range(const size_t var, const size_t bdd_index) const;
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

            std::vector<size_t> nr_bdds_per_variable_;
            std::vector<size_t> bdds_of_variable_offset_;
            two_dim_variable_array<size_t> bdd_of_variable_bdd_index_offset_;
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
        std::vector<size_t> nr_bdd_nodes_per_variable(bdd_storage_.nr_variables(), 0);
        std::vector<size_t> nr_bdds_per_variable(bdd_storage_.nr_variables(), 0);

        // TODO: std::unordered_set might be faster (or tsl::unordered_set)
        class var_cover {
            public:
                std::vector<char> variable_counted;//(bdd_storage_.nr_variables(), 0);
                std::vector<size_t> variables_covered;

                var_cover(const size_t n)
                    : variable_counted(n, 0)
                {}
                bool variable_covered(const size_t v) 
                {
                    assert(v < variable_counted.size());
                    assert(variable_counted[v] == 0 || variable_counted[v] == 1);
                    return variable_counted[v] == 1;
                }
                void cover_variable(const size_t v)
                {
                    assert(!variable_covered(v));
                    variable_counted[v] = 1;
                    variables_covered.push_back(v);
                }
                void uncover_variables()
                {
                    for(const size_t v : variables_covered) {
                        assert(variable_covered(v));
                        variable_counted[v] = 0;
                    }
                    variables_covered.clear(); 
                }
        };

//#pragma omp parallel
        {
            var_cover vc(bdd_storage_.nr_variables());
//#pragma omp for reduction(vec_size_t_plus:nr_bdds_per_variable)
            for(size_t bdd_index=0; bdd_index<bdd_storage_.bdd_delimiters().size()-1; ++bdd_index) {
                for(size_t i=bdd_storage_.bdd_delimiters()[bdd_index]; i<bdd_storage_.bdd_delimiters()[bdd_index+1]; ++i) {
                    const size_t bdd_variable = bdd_storage_.bdd_nodes()[i].variable;
                    ++nr_bdd_nodes_per_variable[bdd_variable];
                    if(!vc.variable_covered(bdd_variable)) {
                        vc.cover_variable(bdd_variable);
                        ++nr_bdds_per_variable[bdd_variable];
                    }
                }
                vc.uncover_variables();
            }
        }
        assert(bdd_storage_.bdd_nodes().size() == std::accumulate(nr_bdd_nodes_per_variable.begin(), nr_bdd_nodes_per_variable.end(), 0));
        
        std::vector<size_t> bdd_offset_per_variable;
        bdd_offset_per_variable.reserve(bdd_storage_.nr_variables());
        bdd_offset_per_variable.push_back(0);
        std::partial_sum(nr_bdd_nodes_per_variable.begin(), nr_bdd_nodes_per_variable.end()-1, std::back_inserter(bdd_offset_per_variable));
        assert(bdd_offset_per_variable.size() == bdd_storage_.nr_variables());
        assert(bdd_offset_per_variable.back() + nr_bdd_nodes_per_variable.back() == bdd_storage_.bdd_nodes().size());

        bdd_branch_nodes_.resize(bdd_storage_.bdd_nodes().size());
        bdd_variables_.resize(nr_bdds_per_variable.begin(), nr_bdds_per_variable.end());

        // fill branch instructions into datastructures and set pointers
        std::fill(nr_bdd_nodes_per_variable.begin(), nr_bdd_nodes_per_variable.end(), 0); // counter for bdd instruction per variable
        std::fill(nr_bdds_per_variable.begin(), nr_bdds_per_variable.end(), 0); // counter for bdd per variable

//#pragma omp parallel
        {
            tsl::robin_map<size_t, BDD_BRANCH_NODE*> stored_bdd_node_index_to_bdd_address;
            var_cover vc(bdd_storage_.nr_variables());
            size_t c = 0; // check if everything is read contiguously
//#pragma omp for reduction(vec_size_t_plus:nr_bdds_per_variable)
            for(size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index) {
                ++nr_bdds_;
                vc.uncover_variables(); 
                stored_bdd_node_index_to_bdd_address.clear();
                stored_bdd_node_index_to_bdd_address.insert(std::pair<size_t, BDD_BRANCH_NODE*>(bdd_storage::bdd_node::terminal_0, BDD_BRANCH_NODE::terminal_0()));
                stored_bdd_node_index_to_bdd_address.insert(std::pair<size_t, BDD_BRANCH_NODE*>(bdd_storage::bdd_node::terminal_1, BDD_BRANCH_NODE::terminal_1()));
                BDD_VARIABLE* next_bdd_var = nullptr;

                //std::cout << "bdd index = " << bdd_index << "\n";
                const size_t first_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index]; 
                const size_t last_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index+1];
                //std::cout << "bdd delimiter = " << bdd_storage_.bdd_delimiters()[bdd_index+1] << "\n";
                for(size_t stored_bdd_node_index=first_stored_bdd_node; stored_bdd_node_index<last_stored_bdd_node; ++stored_bdd_node_index, ++c) {
                    assert(c == stored_bdd_node_index);
                    //std::cout << "stored bdd node index = " << stored_bdd_node_index << "\n";
                    const auto& stored_bdd = bdd_storage_.bdd_nodes()[stored_bdd_node_index];
                    const size_t v = stored_bdd.variable;
                    //std::cout << "bdd variable = " << v << ", bdd variable offset = " << bdd_offset_per_variable[v] << "\n";

                    auto& bdd_var = bdd_variables_(v, nr_bdds_per_variable[v]);
                    if(!vc.variable_covered(v)) {
                        // assert(bdd_var.is_initial_state());
                        vc.cover_variable(v);

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

                    const size_t bdd_branch_nodes_index = bdd_var.last_node_index; 
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

                for(const size_t v : vc.variables_covered) {
                    ++nr_bdds_per_variable[v];
                }
            }
        }

        for(size_t v=0; v<nr_bdds_per_variable.size(); ++v) {
            assert(nr_bdds_per_variable[v] == bdd_variables_[v].size());
        }

        for(const auto& bdd : bdd_branch_nodes_) {
            check_bdd_branch_node(bdd);
        }

        // set variable indices if supported
        if constexpr(has_variable_indices<BDD_VARIABLE>::value)
        {
            std::cout << "Set variable indices\n";
#pragma omp parallel for
            for(size_t i=0; i<this->nr_variables(); ++i)
            {
                for(size_t bdd_index=0; bdd_index<nr_bdds(i); ++bdd_index) {
                    {
                        bdd_variables_(i, bdd_index).variable = i;
                    }
                } 
            }

            for(size_t i=0; i<this->nr_variables(); ++i)
            {
                for(size_t bdd_index=0; bdd_index<nr_bdds(i); ++bdd_index)
                {
                    auto& bdd_var = bdd_variables_(i, bdd_index);
                    if(bdd_var.is_first_bdd_variable())
                        bdd_var.first_variable = i;
                    else
                        bdd_var.first_variable = bdd_var.prev->first_variable;
                }
            }

            for(std::ptrdiff_t i=this->nr_variables()-1; i>=0; --i)
            {
                for(size_t bdd_index=0; bdd_index<nr_bdds(i); ++bdd_index)
                {
                    auto& bdd_var = bdd_variables_(i, bdd_index);
                    if(bdd_var.is_last_bdd_variable())
                        bdd_var.last_variable = i;
                    else
                        bdd_var.last_variable = bdd_var.next->last_variable;
                }
            }
        }

        for(size_t v=0; v<nr_variables(); ++v) {
            for(size_t bdd_index=0; bdd_index<nr_bdds(v); ++bdd_index) {
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
    size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_branch_node_index(const BDD_BRANCH_NODE* bdd) const
    {
        assert(bdd >= &bdd_branch_nodes_[0]);
        const size_t i = bdd - &bdd_branch_nodes_[0];
        assert(i < bdd_branch_nodes_.size());
        return i; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::variable_index(const BDD_BRANCH_NODE& bdd) const
    {
        const size_t i = bdd_branch_node_index(&bdd);

        size_t lb = 0;
        size_t ub = nr_variables()-1;
        size_t v = nr_variables()/2;
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
    size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::variable_index(const BDD_VARIABLE& bdd_var) const
    {
        assert(&bdd_var >= &bdd_variables_(0,0));
        assert(&bdd_var <= &bdd_variables_.back().back());
        size_t lb = 0;
        size_t ub = nr_variables()-1;
        size_t v = nr_variables()/2;
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
    size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::first_variable_of_bdd(const BDD_VARIABLE& bdd_var) const
    {
        BDD_VARIABLE const* p = &bdd_var;
        while(p->prev != nullptr)
            p = p->prev;
        return variable_index(*p);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    size_t bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::last_variable_of_bdd(const BDD_VARIABLE& bdd_var) const
    {
        BDD_VARIABLE const* p = &bdd_var;
        while(p->next != nullptr)
            p = p->next;
        return variable_index(*p); 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    bool bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::last_variable_of_bdd(const size_t var, const size_t bdd_index) const
    {
        const BDD_VARIABLE& bdd_var = bdd_variables_(var, bdd_index);
        return bdd_var.next == nullptr; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    bool bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::first_variable_of_bdd(const size_t var, const size_t bdd_index) const
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
        const size_t first_node_index = bdd_var.first_node_index;
        const size_t last_node_index = bdd_var.last_node_index;
        for(size_t i=first_node_index; i<last_node_index; ++i) {
            const auto& node = bdd_branch_nodes_[i];
            if(node.low_outgoing != node.terminal_0())
                ++n;
            if(node.high_outgoing != node.terminal_0())
                ++n;
        }

        return n;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_step(const BDD_VARIABLE& bdd_var)
    {
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
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_step(const size_t var, const size_t bdd_index)
    {
        assert(var < this->nr_variables());
        assert(bdd_index < this->nr_bdds(var));

        const auto& bdd_var = get_bdd_variable(var, bdd_index);
        forward_step(bdd_var); 
    }


    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_step(const size_t var)
    {
        assert(var < bdd_variables_.size());
        for(size_t bdd_index = 0; bdd_index<bdd_variables_[var].size(); ++bdd_index)
            forward_step(var, bdd_index);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::backward_step(const size_t var, const size_t bdd_index)
    {
        assert(var < bdd_variables_.size());
        assert(bdd_index < bdd_variables_[var].size());

        auto& bdd_var = bdd_variables_(var,bdd_index);
        assert(var+1 != nr_variables() || bdd_var.next == nullptr);
        backward_step(bdd_var);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::backward_step(const BDD_VARIABLE& bdd_var)
    {
        // iterate over all bdd nodes and make forward step
        const std::ptrdiff_t first_node_index = bdd_var.first_node_index;
        const std::ptrdiff_t last_node_index = bdd_var.last_node_index;
        for(std::ptrdiff_t i=last_node_index-1; i>=first_node_index; --i)
            bdd_branch_nodes_[i].backward_step();
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::backward_step(const size_t var)
    {
        assert(var < bdd_variables_.size());
        for(size_t bdd_index = 0; bdd_index<bdd_variables_[var].size(); ++bdd_index)
            backward_step(var, bdd_index);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::backward_run()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        for(std::ptrdiff_t i=this->bdd_branch_nodes_.size()-1; i>=0; --i)
        {
            this->bdd_branch_nodes_[i].backward_step();
        }
        //for(std::ptrdiff_t var=nr_variables()-1; var>=0; --var)
        //    backward_step(var); 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::forward_run()
    {
        for(size_t var=0; var<nr_variables(); ++var)
            forward_step(var); 
    }

    template <typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    BDD_VARIABLE &bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::get_bdd_variable(const size_t var, const size_t bdd_index)
    {
        assert(var < nr_variables());
        assert(bdd_index < bdd_variables_[var].size());
        return bdd_variables_(var, bdd_index);
    }

    template <typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    const BDD_VARIABLE &bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::get_bdd_variable(const size_t var, const size_t bdd_index) const
    {
        assert(var < nr_variables());
        assert(bdd_index < bdd_variables_[var].size());
        return bdd_variables_(var, bdd_index);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    const BDD_BRANCH_NODE& bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::get_bdd_branch_node(const size_t var, const size_t bdd_index, const size_t bdd_node_index) const
    {
        assert(var < nr_variables());
        assert(bdd_index < bdd_variables_[var].size());
        assert(bdd_node_index < bdd_variables_(var,bdd_index).nr_bdd_nodes());
        return bdd_branch_nodes_[ bdd_variables_(var,bdd_index).first_node_index + bdd_node_index ];
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        template<typename ITERATOR>
        bool bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::check_feasibility(ITERATOR var_begin, ITERATOR var_end) const
        {
            assert(std::distance(var_begin, var_end) == this->nr_variables());

            std::vector<char> bdd_nbranch_node_marks(this->bdd_branch_nodes_.size(), 0);

            size_t var = 0;
            for(auto var_iter=var_begin; var_iter!=var_end; ++var_iter, ++var) {
                const bool val = *(var_begin+var);
                for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    const auto& bdd_var = this->bdd_variables_(var, bdd_index);
                    if(bdd_var.is_first_bdd_variable()) {
                        bdd_nbranch_node_marks[bdd_var.first_node_index] = 1;
                    }
                    for(size_t bdd_node_index=bdd_var.first_node_index; bdd_node_index<bdd_var.last_node_index; ++bdd_node_index) {
                        if(bdd_nbranch_node_marks[bdd_node_index] == 1) {
                            const auto& bdd = this->bdd_branch_nodes_[bdd_node_index];
                            const auto* bdd_next_index = [&]() {
                                if(val == false)
                                    return bdd.low_outgoing;
                                else 
                                    return bdd.high_outgoing;
                            }();

                            if(bdd_next_index == BDD_BRANCH_NODE::terminal_0())
                                return false;
                            if(bdd_next_index == BDD_BRANCH_NODE::terminal_1()) {
                            } else { 
                                bdd_nbranch_node_marks[ this->bdd_branch_node_index(bdd_next_index) ] = 1;
                            }
                        }
                    }
                }
            }

            return true;
        }

}
