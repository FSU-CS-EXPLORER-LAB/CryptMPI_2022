/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */

#ifndef __MYLIST_H__
#define __MYLIST_H__

struct list_head {
    struct list_head *next, *prev;
};

static inline void MV2_INIT_LIST_HEAD(struct list_head *list)
{
    list->next = list;
    list->prev = list;
}

#define mv2_offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define mv2_container_of(ptr, type, member) ({        \
     const typeof( ((type *)0)->member ) *__mptr = (ptr);    \
     (type *)( (char *)__mptr - mv2_offsetof(type,member) );})

#define mv2_list_entry(ptr, type, member) \
    mv2_container_of(ptr, type, member)

#define mv2_list_for_each(pos, head) \
    for (pos = (head)->next; pos != (head); pos = pos->next)

#define mv2_list_for_each_safe(pos, n, head) \
     for (pos = (head)->next, n = pos->next; pos != (head); \
           pos = n, n = pos->next)

#define mv2_list_for_each_entry(pos, head, member)        \
      for (pos = mv2_list_entry((head)->next, typeof(*pos), member);      \
        &pos->member != (head);        \
        pos = mv2_list_entry(pos->member.next, typeof(*pos), member))

#define mv2_list_for_each_entry_safe(pos, n, head, member)                  \
  for (pos = mv2_list_entry((head)->next, typeof(*pos), member),      \
          n = mv2_list_entry(pos->member.next, typeof(*pos), member); \
          &pos->member != (head);                                    \
          pos = n, n = mv2_list_entry(n->member.next, typeof(*n), member))

///     insert / delete entry from a list

static inline void mv2_imp_list_add(struct list_head *new, struct list_head *prev, struct list_head *next)
{
    next->prev = new;
    new->next = next;
    new->prev = prev;
    prev->next = new;
}

static inline void mv2_list_add(struct list_head *new, struct list_head *head)
{
    mv2_imp_list_add(new, head, head->next);
}

static inline void mv2_list_add_tail(struct list_head *new, struct list_head *head)
{
    mv2_imp_list_add(new, head->prev, head);
}

static inline void mv2_imp_list_del(struct list_head *prev, struct list_head *next)
{
    next->prev = prev;
    prev->next = next;
}

static inline void mv2_list_del(struct list_head *entry)
{
    mv2_imp_list_del(entry->prev, entry->next);
}

static inline int mv2_list_empty(const struct list_head *head)
{
    return head->next == head;
}

#endif
