/**
 * igmp_group.c - 组成员表管理模块
 * 实验三：IGMP协议软件（路由器端）开发
 */

#include "igmp.h"

/**
 * 初始化组成员表
 */
void group_table_init(void)
{
    g_group_list = NULL;
    printf("[IGMP] 组成员表已初始化\n");
}

/**
 * 清理组成员表
 */
void group_table_cleanup(void)
{
    struct group_entry *entry, *next;

    entry = g_group_list;
    while (entry) {
        next = entry->next;
        free(entry);
        entry = next;
    }
    g_group_list = NULL;
}

/**
 * 查找组记录
 */
struct group_entry* group_find(uint32_t addr)
{
    struct group_entry *entry;

    for (entry = g_group_list; entry != NULL; entry = entry->next) {
        if (entry->group_addr == addr) {
            return entry;
        }
    }
    return NULL;
}

/**
 * 添加组记录
 */
int group_add(uint32_t addr)
{
    struct group_entry *entry;
    struct in_addr in;

    /* 检查是否已存在 */
    if (group_find(addr) != NULL) {
        return 0;  /* 已存在 */
    }

    /* 检查容量 */
    if (group_count() >= MAX_GROUPS) {
        fprintf(stderr, "[ERROR] 组成员表已满\n");
        return -1;
    }

    /* 分配新记录 */
    entry = (struct group_entry *)malloc(sizeof(struct group_entry));
    if (entry == NULL) {
        perror("malloc");
        return -1;
    }

    /* 初始化 */
    entry->group_addr = addr;
    in.s_addr = addr;
    strncpy(entry->group_str, inet_ntoa(in), sizeof(entry->group_str) - 1);
    entry->expire_time = time(NULL) + IGMP_GROUP_MEMBERSHIP_INTERVAL;
    entry->query_count = 0;
    entry->next = g_group_list;
    g_group_list = entry;

    return 1;  /* 新添加 */
}

/**
 * 删除组记录
 */
int group_remove(uint32_t addr)
{
    struct group_entry *entry, *prev = NULL;

    for (entry = g_group_list; entry != NULL; prev = entry, entry = entry->next) {
        if (entry->group_addr == addr) {
            if (prev) {
                prev->next = entry->next;
            } else {
                g_group_list = entry->next;
            }
            printf("[GROUP] 删除组: %s\n", entry->group_str);
            free(entry);
            return 1;
        }
    }
    return 0;  /* 未找到 */
}

/**
 * 更新组超时时间
 */
void group_update_timer(uint32_t addr)
{
    struct group_entry *entry = group_find(addr);
    if (entry) {
        entry->expire_time = time(NULL) + IGMP_GROUP_MEMBERSHIP_INTERVAL;
    }
}

/**
 * 检查并删除超时的组
 */
void group_check_expire(void)
{
    struct group_entry *entry, *next, *prev = NULL;
    time_t now = time(NULL);

    entry = g_group_list;
    while (entry) {
        next = entry->next;
        
        if (now >= entry->expire_time) {
            printf("[EXPIRE] 组 %s 超时，已删除\n", entry->group_str);
            
            if (prev) {
                prev->next = next;
            } else {
                g_group_list = next;
            }
            free(entry);
            entry = next;
            continue;
        }
        
        prev = entry;
        entry = next;
    }
}

/**
 * 获取组数量
 */
int group_count(void)
{
    int count = 0;
    struct group_entry *entry;

    for (entry = g_group_list; entry != NULL; entry = entry->next) {
        count++;
    }
    return count;
}

/**
 * 打印所有组成员
 */
void group_print_all(void)
{
    struct group_entry *entry;
    int i = 0;
    time_t now = time(NULL);

    printf("┌────────────────────────────────────────┐\n");
    printf("│         当前组播组成员表               │\n");
    printf("├────┬─────────────────┬─────────────────┤\n");
    printf("│ #  │ 组播组地址      │ 剩余时间(秒)    │\n");
    printf("├────┼─────────────────┼─────────────────┤\n");

    for (entry = g_group_list; entry != NULL; entry = entry->next) {
        i++;
        long remain = entry->expire_time - now;
        if (remain < 0) remain = 0;
        printf("│ %-2d │ %-15s │ %-15ld │\n",
               i, entry->group_str, remain);
    }

    if (i == 0) {
        printf("│         (无组播组成员)                │\n");
    }

    printf("└────┴─────────────────┴─────────────────┘\n");
    printf("  共 %d 个组播组\n", i);
}
