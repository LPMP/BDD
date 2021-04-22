#include "avl_tree.hxx"
#include <string>
#include "test.h"

using namespace LPMP;

struct data_type
{
    std::string name;
    int lb_;
    int ub_;
    avl_node<data_type>* wrapper_;
};

int main(int argc, char** argv)
{
    data_type t1;
    t1.lb_ = 257;
    t1.ub_ = 261;
    t1.name = "t1";
    data_type t2;
    t2.lb_ = 233;
    t2.ub_ = 236;
    t2.name = "t2";
    data_type t3;
    t3.lb_ = 277;
    t3.ub_ = 320;
    t3.name = "t3";
    data_type t4;
    t4.lb_ = 189;
    t4.ub_ = 192;
    t4.name = "t4";
    data_type t5;
    t5.lb_ = 213;
    t5.ub_ = 217;
    t5.name = "t5";

    avl_tree<data_type> tree;

    data_type * ptr = tree.create_node(t1);
    tree.insert(ptr->wrapper_);
    ptr = tree.create_node(t2);
    tree.insert(ptr->wrapper_);
    ptr = tree.create_node(t3);
    tree.insert(ptr->wrapper_);
    ptr = tree.create_node(t4);
    tree.insert(ptr->wrapper_);

    tree.write();

    ptr = tree.create_node(t5);
    tree.insert(ptr->wrapper_);

    tree.write();
}
