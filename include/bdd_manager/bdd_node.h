#pragma once

#include <random>
#include <cassert>
#include <vector>
#include <functional>

namespace BDD {

constexpr static std::size_t logvarsize = 41;
constexpr static std::size_t maxvarsize = static_cast<std::size_t>(1) << logvarsize;
constexpr static std::size_t unique_table_hash_size = 22;
constexpr static std::size_t hashtablesize = static_cast<std::size_t>(1) << unique_table_hash_size;

class bdd_mgr;

class node_struct
{
    public:
    //~node_struct();
    //node_struct(node_struct&& o);

    void init_new_node(std::size_t v, node_struct* l, node_struct* h);
    std::size_t hash_code() const;
    void mark(); // TODO: make non-recursive and add recursive marking as well
    void unmark(); // TODO: make non-recursive and add recursive marking as well
    bool marked() const;

    void recursively_revive();
    void recursively_kill();
    void deref();
    bool dead() const { return xref <= 0; }

    template<typename ITERATOR>
    bool evaluate(ITERATOR var_begin, ITERATOR var_end);

    size_t nr_nodes();
    std::vector<node_struct*> nodes_postorder();
    std::vector<node_struct*> nodes_bfs();
    std::vector<size_t> variables();
    bool exactly_one_solution();

    void init_botsink(bdd_mgr* mgr);
    bool is_botsink() const;
    void init_topsink(bdd_mgr* mgr);
    bool is_topsink() const;
    bool is_terminal() const { return is_topsink() || is_botsink(); }

    union { node_struct* lo; bdd_mgr* bdd_mgr_1; node_struct* next_available; };
    union { node_struct* hi; bdd_mgr* bdd_mgr_2; };
	//node_struct* lo = nullptr;
    //node_struct* hi = nullptr;
    // TODO: make enum
    //node_struct* next_available = nullptr;
    
    // TODO: possibly put away with hash_key field and combine index, marked and xref into one 64 bit field.
    
    std::size_t index : logvarsize;
    std::size_t hash_key : unique_table_hash_size; // make const
    std::size_t marked_ : 1;
    //std::size_t large_subtree : 1; // subtree is large enough so that recursively visiting nodes would exceed stack
	int xref = 0;

    constexpr static size_t botsink_index = std::pow(2,logvarsize)-1;
    constexpr static size_t topsink_index = std::pow(2,logvarsize)-2;

    template<typename STREAM>
        void print(STREAM& s);

    bdd_mgr* find_bdd_mgr();

    private:
    size_t nr_nodes_impl();
    void variables_impl(std::vector<size_t>&);
    void nodes_postorder_impl(std::vector<node_struct*>&);
    // depth first search bdd to find terminal node, where link to bdd mgr is stored

    template<typename STREAM>
        void print_rec(STREAM& s);
};

using node = node_struct;

class node_ref {
    public:
    node_ref(const node_ref& o);
    node_ref(node* r);
    ~node_ref();
    node_ref(node_ref&& r);
    node_ref();
    node* address() const { return ref; }
    bdd_mgr& get_bdd_mgr() const;
    node_ref low() { return node_ref(ref->lo); }
    node_ref high() { return node_ref(ref->hi); }
    bool is_botsink() const { return ref->is_botsink(); }
    bool is_topsink() const { return ref->is_topsink(); }
    bool is_terminal() const { return ref->is_terminal(); }
    size_t nr_nodes() const { return ref->nr_nodes(); }
    bool exactly_one_solution() const { return ref->exactly_one_solution(); }
    bdd_mgr* find_bdd_mgr() { return ref->find_bdd_mgr(); }
    node_ref botsink();
    node_ref topsink();

    size_t reference_count() const { return ref->xref; }

    template<typename ITERATOR>
    bool evaluate(ITERATOR var_begin, ITERATOR var_end) { return ref->evaluate(var_begin, var_end); }

    size_t variable() const { return ref->index; }
    std::vector<size_t> variables() { return ref->variables(); }

    template<typename STREAM>
        void print(STREAM& s) { return ref->print(s); }

    friend class bdd_mgr;

    bool operator==(const node_ref& o) const { return ref == o.ref; }
    bool operator!=(const node_ref& o) const { return !(ref == o.ref); }
    node_ref& operator=(const node_ref& o);

    std::vector<node_ref> nodes_postorder();
    std::vector<node_ref> nodes_bfs();

    bool marked() const { return ref->marked_; }
    void mark() { ref->marked_ = 1; }
    void unmark() { ref->marked_ = 0; }
    void unmark_rec() { ref->unmark(); }

    private:
    node* ref = nullptr;
};

template<typename ITERATOR>
bool node_struct::evaluate(ITERATOR var_begin, ITERATOR var_end)
{
    if(this->is_botsink())
        return false;
    if(this->is_topsink())
        return true;
    assert(index < std::distance(var_begin, var_end));
    const bool x = *(var_begin + this->index);
    if(x == true)
        return hi->evaluate(var_begin, var_end);
    else
        return lo->evaluate(var_begin, var_end);

}

template<typename STREAM>
void node_struct::print(STREAM& s)
{
    s << "digraph BDD {\n";
    print_rec(s);
    unmark();
    s << "}\n";
}

template<typename STREAM>
void node_struct::print_rec(STREAM& s)
{
    if(is_terminal())
        return;

    assert(marked_ == 0);
    marked_ = 1;
    auto node_id = [](node* p) -> std::string {
        if(p->is_botsink())
            return std::string("bot");
        if(p->is_topsink())
            return std::string("top");
        return std::string("\"") + std::to_string(p->index) + "," + std::to_string(size_t(p)) + "\"";
    };

    s << node_id(this) << " -> " << node_id(lo) << " [label=\"0\"]\n";
    s << node_id(this) << " -> " << node_id(hi) << " [label=\"1\"]\n";

    if(lo->marked_ == 0)
        lo->print_rec(s);
    if(hi->marked_ == 0)
        hi->print_rec(s);
}

}
// insert hsh function for node_ref
namespace std {

    template<>
        struct hash<BDD::node_ref>
        {
            size_t operator()(const BDD::node_ref& x) const
            {
                return hash<BDD::node*>()(x.address());
            }

        };
}
