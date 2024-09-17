#include <stdio.h>
#include <stdlib.h>

int main(void) {
    FILE* file = fopen("data/tiny_shakespear.txt", "r");
    if (file == NULL) {
        perror("Failed to open file");
        return 1;
    }

    int buffer_size = 1024;
    char buffer[buffer_size];
    while (fgets(buffer, buffer_size, file) != NULL) {
        printf("%s", buffer);
    }

    fclose(file);
	return 0;
}