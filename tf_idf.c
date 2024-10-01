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
            return;
        }
        i += 1;
        current = &((*current)->next);
    }

    // Node was not found, insert a new one.
    HashNode* node = (HashNode*) malloc(sizeof(HashNode));
    node->key = malloc(strlen(key) + 1);  // +1 to account for '\0'.
    strcpy(node->key, key);
    node->value = value;
    node->next = NULL;
    *current = node;
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


typedef struct {
    HashMap* map;
    int idx;
    HashNode* next;

} HashMapIterator;


HashMapIterator* HashMapIterator_create(HashMap* map) {
    HashMapIterator* iter = (HashMapIterator*)malloc(sizeof(HashMapIterator));
    iter->map = map;
    iter->idx = -1;
    iter->next = NULL;
    return iter;
}


HashNode* HashMapIterator_next(HashMapIterator* iter) {
    // First try to get the next item in the hash bucket.
    if (iter->next && iter->next->next) {
        iter->next = iter->next->next;
        return iter->next;
    }

    // If no such item exists, keep iterating over the hash buckets
    // to find the next item.
    iter->idx += 1;
    while (iter->idx <= iter->map->max_size_ && !iter->map->nodes[iter->idx]) {
        iter->idx += 1;
    }

    // If we went outside the HashMap, this means there were no more items
    // and we just return NULL.
    if (iter->idx >= iter->map->max_size_) {
        return NULL;
    }

    // Otherwise return the next item we found.
    iter->next = iter->map->nodes[iter->idx];
    return iter->next;
}


typedef struct {
    int* idxs;
    float* values;
    int size;
    int max_size_;
} SparseVector;


SparseVector* SparseVector_create(int max_size) {
    int* idxs = (int*)malloc(max_size * sizeof(int));
    float* values = (float*)malloc(max_size * sizeof(float));
    SparseVector* vec = (SparseVector*)malloc(sizeof(SparseVector));
    vec->idxs= idxs;
    vec->values = values;
    vec->size = 0;
    vec->max_size_ = max_size;
    return vec;
}


void SparseVector_push(SparseVector* vec, int idx, float value) {
    // Increase the vector size if needed.
    if (vec->size == vec->max_size_) {
        int new_max_size = 2 * vec->max_size_;
        vec->idxs = realloc(vec->idxs, new_max_size * sizeof(int));
        vec->values = realloc(vec->values, new_max_size * sizeof(float));
        vec->max_size_ = new_max_size;
    }
    vec->idxs[vec->size] = idx;
    vec->values[vec->size] = value;
    vec->size += 1;
}


float dot(SparseVector* left, SparseVector* right) {
    float result = 0.0;
    int left_idx = 0, right_idx = 0;
    while (left_idx < left->size && right_idx < right->size) {
        if (left->idxs[left_idx] == right->idxs[right_idx]) {
            result += left->values[left_idx] * right->values[right_idx];
            left_idx += 1; right_idx += 1;
        } else if (left->idxs[left_idx] < right->idxs[right_idx]) {
            left_idx += 1;
        } else {
            right_idx += 1;
        }
    }
    return result;
}


typedef struct {
    int idx;
    float score;
} Score;


int score_cmp(const void* left, const void* right) {
    return -(int)(((Score*)left)->score - ((Score*)right)->score);
}


// Scores the query against all documents.
// score_buffer is expected to be an array of Scores with size n_documents.
// When the function returns, Scores will be sorted by dot(query, document) score.
void score_and_sort(
    SparseVector* query, 
    SparseVector* documents[], 
    int n_documents, 
    Score* score_buffer) {
    for (int i = 0; i < n_documents; i++) {
        score_buffer[i].idx = i; 
        score_buffer[i].score = dot(query, documents[i]);
    }
    qsort(score_buffer, n_documents, sizeof(Score), score_cmp);
}


int main(void) {
    FILE* file = fopen("data/tiny_shakespear.txt", "r");
    if (file == NULL) {
        perror("Failed to open file");
        return 1;
    }

    // Create vocabulary.
    HashMap* vocabulary = HashMap_create(32768);
    int buffer_size = 1024;
    char buffer[buffer_size];
    while (fgets(buffer, buffer_size, file)) {
        char* token; 
        token = strtok(buffer, " ");
        while (token != NULL) {
            if (HashMap_get(vocabulary, token) < 0) {
                HashMap_insert(vocabulary, token, vocabulary->size);
            }
            token = strtok(NULL, " ");
        }
    }
    printf("Vocabulary size: %d\n", vocabulary->size);


    // Chunk and embed text. 
    // For now, each line is a separate chunk.
    rewind(file);
    int n_chunks = 40000;
    HashMap* vectors[n_chunks];
    int i = 0;
    while (fgets(buffer, buffer_size, file)) {
        HashMap* vec = HashMap_create(32);

        char* token; 
        token = strtok(buffer, " ");
        while (token) {
            int idx = HashMap_get(vocabulary, token);
            if (idx >= 0) {
                HashMap_insert(vec, token, 1);
            }
            token = strtok(NULL, " ");
        }

        vectors[i] = vec;
        i += 1;
    }
    fclose(file);

	return 0;
}