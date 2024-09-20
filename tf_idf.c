#include <stdio.h>
#include <stdlib.h>


unsigned long djb2(char* str) {
    unsigned long hash = 5381;
    int c;

    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 32 + c
    }
    return hash;
}


int main(void) {
    FILE* file = fopen("data/tiny_shakespear.txt", "r");
    if (file == NULL) {
        perror("Failed to open file");
        return 1;
    }

    int buffer_size = 1024;
    char buffer[buffer_size];
    while (fgets(buffer, buffer_size, file) != NULL) {
        if (buffer[0] != '\n') {
            printf("%lu\n", djb2(buffer) % 1000);
        }
    }

    fclose(file);
	return 0;
}