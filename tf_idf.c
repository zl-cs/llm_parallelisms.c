#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

unsigned long djb2(char* str) {
    unsigned long hash = 5381;
    int c;

    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 32 + c
    }
    return hash;
}


typedef struct HashNode {
    char* key;
    int value;
    struct HashNode* next; 
} HashNode;


typedef struct HashMap {
    HashNode** nodes;
    int size; 
    int max_size_;
} HashMap;


HashMap* HashMap_create(int max_size) {
    HashNode** nodes = (HashNode**)calloc(max_size, sizeof(HashNode*));
    HashMap* map = (HashMap*)malloc(sizeof(HashMap));
    map->nodes = nodes;
    map->size = 0;
    map->max_size_ = max_size;
    return map;
}


void HashMap_insert(HashMap* map, char* key, int value) {
    int idx = djb2(key) % map->max_size_;

    int i = 0;
    // Update value if current node already exists.
    HashNode** current = &map->nodes[idx];
    while (*current) {
        if (strcmp((*current)->key, key) == 0) {
            (*current)->value = value;
            // printf("found node %d as (%d, %d)\n", map->size, idx, i);
            return;
        }
        i += 1;
        current = &((*current)->next);
    }

    // Node was not found, insert a new one.
    HashNode* node = (HashNode*) malloc(sizeof(HashNode));
    node->key = malloc(strlen(key) + 1);
    strcpy(node->key, key);
    node->value = value;
    node->next = NULL;
    *current = node;
    // printf("insert node %d as (%d, %d)\n", map->size, idx, i);
    map->size += 1;
}


int HashMap_get(HashMap* map, char* key) {
    int idx = djb2(key) % map->max_size_;

    HashNode* current = map->nodes[idx];
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next; 
    }
    return -1; 
}


int main(void) {

    FILE* file = fopen("data/tiny_shakespear.txt", "r");
    if (file == NULL) {
        perror("Failed to open file");
        return 1;
    }

    // Tokenize file.
    HashMap* map = HashMap_create(32768);
    int buffer_size = 1024;
    char buffer[buffer_size];
    int i = 0;
    while (fgets(buffer, buffer_size, file)) {
        HashMap_insert(map, buffer, 1);
        char *token; 
        token = strtok(buffer, " ");
        while (token != NULL) {
            HashMap_insert(map, token, 1);
            token = strtok(NULL, " ");
        }
    }

    printf("Map size: %d\n", map->size);

    fclose(file);
	return 0;
}