#include "bdd_manager/bdd_node.h"
#include "bdd_manager/bdd_mgr.h"
#include <cassert>
#include <algorithm>
#include <cstring>
#include <deque>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <iostream> // TODO: remove

namespace BDD {

    void node_struct::init_new_node(std::size_t v, node_struct* l, node_struct* h) 
    {
        assert(v < std::pow(2,logvarsize));
        lo = l;
        hi = h;
        hi->xref++;
        lo->xref++;
        index = v;
        marked_ = 0;
        xref = 0;

        static std::random_device rd;
        static std::mt19937 unique_table_gen(rd());
        static std::uniform_int_distribution<std::size_t> unique_table_distribution(0,hashtablesize-1);

        hash_key = unique_table_distribution(unique_table_gen);

    }

    size_t node::nr_nodes()
    {
        assert(marked_ == 0);
        const size_t n = nr_nodes_impl();
        unmark();
        return n;
    }

    size_t node::nr_nodes_impl()
    {
        // TODO: make non-recursive
        if(is_terminal())
            return 0;
        assert(marked_ == 0);

        size_t n = 1;
        marked_ = 1;
        if(lo->marked_ == 0)
            n += lo->nr_nodes_impl();
        if(hi->marked_ == 0)
            n += hi->nr_nodes_impl();
        
        return n;
    }

    std::vector<node*> node::nodes_postorder()
    {
        assert(marked_ == 0);
        std::vector<node*> n;
        nodes_postorder_impl(n);
        unmark();
        assert(n.size() == nr_nodes());
        return n;
    }

    void node::nodes_postorder_impl(std::vector<node*>& n)
    {
        if(is_terminal())
            return;
        assert(marked_ == 0);

        marked_ = 1;
        if(lo->marked_ == 0)
            lo->nodes_postorder_impl(n);
        if(hi->marked_ == 0)
            hi->nodes_postorder_impl(n); 
        n.push_back(this);
    }

    std::vector<node*> node::nodes_bfs()
    {
        assert(marked_ == 0);
        std::vector<node*> nodes;
        std::deque<node*> dq;

        dq.push_back(this);
        this->marked_ = 1;
        while(!dq.empty())
        {
            node* n = dq.front();
            dq.pop_front();
            assert(n->marked_ == 1);
            nodes.push_back(n);

            if(!n->lo->is_terminal() && n->lo->marked_ == 0)
            {
                dq.push_back(n->lo);
                n->lo->marked_ = 1;
            }
            if(!n->hi->is_terminal() && n->hi->marked_ == 0)
            {
                dq.push_back(n->hi);
                n->hi->marked_ = 1;
            }
        }

        unmark();
        assert(nodes.size() == nr_nodes());
        return nodes; 
    }

    size_t node::nr_solutions()
    {
        std::unordered_map<node*, size_t> nr_solutions;
        const std::vector<size_t> vars = variables();
        std::unordered_map<size_t,size_t> var_order;
        for(size_t i=0; i<vars.size(); ++i)
            var_order.insert({vars[i], i});
        var_order.insert({topsink_index, vars.size()});
        var_order.insert({botsink_index, std::numeric_limits<size_t>::max()});
        const size_t nr_sols = nr_solutions_impl(nr_solutions, var_order);
        assert(nr_solutions.count(this) > 0);
        return nr_sols;
    }

    size_t node::nr_solutions_impl(std::unordered_map<node*, size_t>& nr_map, std::unordered_map<size_t, size_t>& var_order)
    {
        auto it = nr_map.find(this);
        if(it != nr_map.end())
        {
            return it->second;
        }
        else if(is_botsink())
        {
            nr_map.insert({this, 0});
            return 0;
        }
        else if(is_topsink())
        {
            nr_map.insert({this, 1});
            return 1;
        }
        else
        {
            const size_t hi_nr = hi->nr_solutions_impl(nr_map, var_order);
            assert(var_order.count(hi->index) > 0);
            const size_t hi_var = var_order.find(hi->index)->second;
            
            const size_t lo_nr = lo->nr_solutions_impl(nr_map, var_order);
            assert(var_order.count(lo->index) > 0);
            const size_t lo_var = var_order.find(lo->index)->second;

            assert(var_order.count(index) > 0);
            const size_t this_var = var_order.find(index)->second;
            assert(hi_var > this_var && lo_var > this_var);

            const size_t nr = 
                (hi_nr > 0 ? std::pow(2,hi_var-this_var-1)*hi_nr : 0)
                +
                (lo_nr > 0 ? std::pow(2,lo_var-this_var-1)*lo_nr : 0);

            nr_map.insert({this, nr});
            return nr;
        }
    }

    void node::init_botsink(bdd_mgr* mgr)
    {
        this->bdd_mgr_1 = mgr;
        this->bdd_mgr_2 = mgr;
        this->xref = 1;
        this->index = botsink_index;
    }

    bool node::is_botsink() const
    {
        assert((this->index == topsink_index || this->index == botsink_index) == (this->lo == this->hi));
        return (this->index == botsink_index);
    }

    void node::init_topsink(bdd_mgr* mgr)
    {
        this->bdd_mgr_1 = mgr;
        this->bdd_mgr_2 = mgr;
        this->xref = 1;
        this->index = topsink_index;

    }

    bool node::is_topsink() const
    {
        assert((this->index == topsink_index || this->index == botsink_index) == (this->lo == this->hi));
        return (this->index == topsink_index);
    }

    std::size_t node_struct::hash_code() const
    {
        assert(lo != nullptr);
        assert(hi != nullptr);
        return (lo->index << 3) ^ (hi->index << 2);
    }

    void node_struct::mark()
    {
        if(is_terminal())
            return;
        if(!marked())
        {
            marked_ = 1;
            lo->mark();
            hi->mark();
        }
    }

    void node_struct::unmark()
    {
        if(is_terminal())
            return;
        if(marked())
        {
            marked_ = 0;
            lo->unmark();
            hi->unmark();
        }
    }

    bool node_struct::marked() const
    {
        return marked_;
    }

    std::vector<size_t> node_struct::variables()
    {
        std::vector<size_t> v;
        variables_impl(v);
        unmark();
        std::sort(v.begin(), v.end());
        v.erase( std::unique(v.begin(), v.end() ), v.end());
        return v; 
    }

    void node_struct::variables_impl(std::vector<size_t>& v)
    {
        if(is_terminal())
            return;
        assert(marked_ == 0);

        marked_ = 1;
        v.push_back(index);
        if(lo->marked_ == 0)
            lo->variables_impl(v);
        if(hi->marked_ == 0)
            hi->variables_impl(v); 
    }

    bool node_struct::exactly_one_solution()
    {
        if(is_topsink())
            return true;
        if(is_botsink())
            return false;
        // there must be exacty one arc that is botsink and one that is not
        if(lo->is_botsink())
            return hi->exactly_one_solution();
        if(hi->is_botsink())
            return lo->exactly_one_solution(); 
        return false; 
    }

    // TODO: make implementations operate without stack
    void node_struct::recursively_revive()
    {
        xref = 0;
        //deadnodes--;
        if(lo->xref<0)
            lo->recursively_revive();
        else 
            lo->xref++;
        if(hi->xref<0)
            hi->recursively_revive();
        else 
            hi->xref++;
    }

    void node_struct::recursively_kill()
    {
        xref= -1;
        //deadnodes++;
        if(lo->xref==0)
            lo->recursively_kill();
        else 
            lo->xref--;
        if(hi->xref==0)
            hi->recursively_kill();
        else 
            hi->xref--;
    }

    void node_struct::deref()
    {
        if(xref == 0) 
            recursively_kill();
        else 
            xref--;
    }

    bdd_mgr* node_struct::find_bdd_mgr()
    {
        if(is_terminal())
            return bdd_mgr_1;
        if(lo->is_terminal())
            return lo->bdd_mgr_1;
        return hi->find_bdd_mgr(); 
    }

    void node_struct::export_graphviz(const std::string& filename)
    {
        const std::string dot_file = std::filesystem::path(filename).replace_extension("dot");
        const std::string png_file = std::filesystem::path(filename).replace_extension("png");
        std::fstream f;
        f.open(dot_file, std::fstream::out | std::ofstream::trunc);
        export_graphviz(f);
        f.close();
        const std::string convert_command = "dot -Tpng " + dot_file + " > " + png_file;
        std::system(convert_command.c_str());
    }

    node_ref::node_ref(node* p)
        : ref(p)
    {
        if(ref != nullptr)
            ref->xref++;
    }

    node_ref::node_ref(const node_ref& o)
        : ref(o.ref)
    {
        if(ref != nullptr)
            ref->xref++;
    }

    node_ref::~node_ref()
    {
        if(ref != nullptr) 
        {
            assert(ref->xref > 0);
            ref->xref--;
        }
    }

    node_ref::node_ref(node_ref&& o)
    {
        std::swap(ref, o.ref);
    }

    node_ref::node_ref()
        : ref(nullptr)
    {}

    node_ref& node_ref::operator=(const node_ref& o)
    { 
        if(ref != nullptr)
            ref->xref--;
        ref = o.ref;
        if(ref != nullptr)
            ref->xref++;
        return *this;
    }

    // Change std::vector<node*> to std::vector<node_ref> in place by byte copying.
    void convert_node_to_node_ref(std::vector<node*>& nodes, std::vector<node_ref>& node_refs)
    {
        std::vector<node*> empty_nodes;
        static_assert(sizeof(std::vector<node*>) == sizeof(std::vector<node_ref>)); // otherwise in place casting will not be possible.
        std::memcpy(&node_refs, &nodes, sizeof(nodes));
        assert(node_refs.size() == nodes.size());
        for(size_t i=0; i<nodes.size(); ++i)
        {
            // this construction is necessary for correct reference counting
            node_ref n(nodes[i]);
            nodes[i] = nullptr;
            node_refs[i] = n;
        }
        std::memcpy(&nodes, &empty_nodes, sizeof(nodes)); 
    }

    std::vector<node_ref> node_ref::nodes_postorder()
    {
        std::vector<node*> nodes = ref->nodes_postorder();
        std::vector<node_ref> node_refs;
        convert_node_to_node_ref(nodes, node_refs);
        return node_refs;
    }

    std::vector<node_ref> node_ref::nodes_bfs()
    {
        std::vector<node*> nodes = ref->nodes_bfs();
        std::vector<node_ref> node_refs;
        convert_node_to_node_ref(nodes, node_refs);
        return node_refs;
    }

    node_ref node_ref::botsink() 
    { 
        return node_ref(find_bdd_mgr()->botsink()); 
    }

    node_ref node_ref::topsink() 
    { 
        return node_ref(find_bdd_mgr()->topsink()); }

}
