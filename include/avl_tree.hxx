#pragma once

#include <list>
#include <cassert>
#include <iostream>

namespace LPMP {

// T : data type that must have the following fields : int lb_, int ub_, avl_node<T> wrapper_
//
template<typename T>
struct avl_node {
    
    T data;
    struct avl_node<T> * left;
    struct avl_node<T> * right;
    int height; // if height == 0, then AVL node is not part of the tree

    int lb() { return data.lb_; }
    int ub() { return data.ub_; }
};


// AVL Tree (without removal), where every node has an associated key range [lb, ub]
//
template<typename T>
class avl_tree {

    public:

        avl_tree()
        {
            root = nullptr;
        }

        T * create_node(T data); // create new AVL node for data
        void insert(avl_node<T> * node_ptr);// insert AVL node into tree (key range must be set prior)
        T * find(int key);
        void write(); // for inspection

        const std::list<avl_node<T>> & get_avl_nodes() const { return nodes; }

    private:

        int height(avl_node<T> * ptr);
        int balance_factor(avl_node<T> * ptr);
        avl_node<T> * rotate_left_left(avl_node<T> * ptr);
        avl_node<T> * rotate_left_right(avl_node<T> * ptr);
        avl_node<T> * rotate_right_left(avl_node<T> * ptr);
        avl_node<T> * rotate_right_right(avl_node<T> * ptr);
        avl_node<T> * insert(avl_node<T> * ptr, avl_node<T> * node_ptr);
        void write(avl_node<T> * ptr);

        avl_node<T> * root;
        std::list<avl_node<T>> nodes; 
};


template<typename T>
int avl_tree<T>::height(avl_node<T> * ptr)
{
    assert(ptr != nullptr);
    if (ptr->left && ptr->right)
    {
        if (ptr->left->height < ptr->right->height)
            return ptr->right->height + 1;
        else
            return ptr->left->height + 1;
    }
    else if (ptr->left && ptr->right == nullptr)
        return ptr->left->height + 1;
    else if (ptr->left == nullptr && ptr->right)
        return ptr->right->height + 1;
    return 0;
}

template<typename T>
int avl_tree<T>::balance_factor(avl_node<T> * ptr)
{
    assert(ptr != nullptr);
    if (ptr->left && ptr->right)
        return ptr->left->height - ptr->right->height; 
    else if (ptr->left && ptr->right == nullptr)
        return ptr->left->height; 
    else if (ptr->left == nullptr && ptr->right)
        return -(ptr->right->height);
}

template<typename T>
avl_node<T> * avl_tree<T>::rotate_left_left(avl_node<T> * ptr)
{
    avl_node<T> * temp;
    avl_node<T> * child;
    temp = ptr;
    child = temp->left;

    temp->left = child->right;
    child->right = temp;

    return child;
}

template<typename T>
avl_node<T> * avl_tree<T>::rotate_left_right(avl_node<T> * ptr)
{
    avl_node<T> * temp;
    avl_node<T> * child;
    avl_node<T> * grandchild;
    temp = ptr;
    child = temp->left;
    grandchild = temp->left->right;

    temp->left = grandchild->right;
    child->right = grandchild->left;
    grandchild->right = temp;
    grandchild->left = child; 
    
    return grandchild;
}

template<typename T>
avl_node<T> * avl_tree<T>::rotate_right_left(avl_node<T> * ptr)
{
    avl_node<T> * temp;
    avl_node<T> * child;
    avl_node<T> * grandchild;
    temp = ptr;
    child = temp->right;
    grandchild = temp->right->left;

    temp->right = grandchild->left;
    child->left = grandchild->right;
    grandchild->left = temp;
    grandchild->right = child; 
    
    return grandchild;   
}

template<typename T>
avl_node<T> * avl_tree<T>::rotate_right_right(avl_node<T> * ptr)
{
    avl_node<T> * temp;
    avl_node<T> * child;
    temp = ptr;
    child = temp->right;

    temp->right = child->left;
    child->left = temp;

    return child;
}


template<typename T>
avl_node<T> * avl_tree<T>::insert(avl_node<T> * ptr, avl_node<T> * node_ptr)
{
    // recursive insertion
    if (ptr == nullptr)
    {
        ptr = node_ptr;
        ptr->left = nullptr;
        ptr->right = nullptr;
        ptr->height = 1;
        return ptr;
    }
    else if (node_ptr->ub() < ptr->lb())
        ptr->left = insert(ptr->left, node_ptr);
    else if (node_ptr->lb() > ptr->ub())
        ptr->right = insert(ptr->right, node_ptr);
    else
    {
        std::cout << "AVL Tree: Key range of inserted data overlaps with existing data (unintended behavior). Abort." << std::endl;
        std::cout << "key range = [" << node_ptr->lb() << "," << node_ptr->ub() << "]" << std::endl;
        write();
        exit(0);
    }

    // rebalancing
    ptr->height = height(ptr);
    if (balance_factor(ptr) == 2 && balance_factor(ptr->left) == 1)
        ptr = rotate_left_left(ptr);
    else if (balance_factor(ptr) == -2 && balance_factor(ptr->right) == -1)
        ptr = rotate_right_right(ptr);
    else if (balance_factor(ptr) == -2 && balance_factor(ptr->right) == 1)
        ptr = rotate_right_left(ptr);
    else if (balance_factor(ptr) == 2 && balance_factor(ptr->left) == -1)
        ptr = rotate_left_right(ptr);

    return ptr;
}

template<typename T>
void avl_tree<T>::insert(avl_node<T> * node_ptr)
{
    // write();
    root = insert(root, node_ptr);
}


template<typename T>
T * avl_tree<T>::create_node(T data)
{
    avl_node<T> node;
    node.data = data;
    nodes.push_back(node);
    avl_node<T> * node_ptr = &nodes.back();
    T * data_ptr = &(node_ptr->data);
    data_ptr->wrapper_ = node_ptr;
    node_ptr->height = 0; // flags that AVL node is not inserted yet

    return data_ptr;
}


template<typename T>
T * avl_tree<T>::find(int key)
{
    avl_node<T> * ptr = root;
    while (ptr != nullptr)
    {
        if (key >= ptr->lb() && key <= ptr->ub())
            return &(ptr->data);
        else if (key < ptr->lb())
            ptr = ptr->left;
        else if (key > ptr->ub())
            ptr = ptr->right;
    }
    return nullptr;
}

template<typename T>
void avl_tree<T>::write()
{
    std::cout << "AVL tree: root" << std::endl;
    write(root);
}

template<typename T>
void avl_tree<T>::write(avl_node<T> * ptr)
{
    if (ptr == nullptr)
        return;

    std::cout << "[" << ptr->lb() << "," << ptr->ub() << "]" << std::endl;
    std::cout << "Left branch: " << std::endl;
    write(ptr->left);
    std::cout << "Right branch: " << std::endl;
    write(ptr->right);
}


} // namespace LPMP
