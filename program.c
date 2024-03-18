#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <SDL2/SDL.h>

//Screen dimension constants.
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

double sqr(double x) {
	return x * x;
}

// Generates a sample from the uniform distribution. 
// Assumes that srand() has already been called.
double uniform() {
	return (double) rand() / RAND_MAX;	
}


// Generates a sample from the standard normal distribution.
// Based on the implementation from Numerical Recipies and 
// https://dl.acm.org/doi/pdf/10.1145/138351.138364.
double normal() {
	double u, v, x, y, q;
	do {
		u = uniform();
		v = 1.7156 * (uniform() - 0.5);
		x = u - 0.449871;
		y = fabs(v) + 0.386595;
		q = sqr(x) + y * (0.19600 * y - 0.25472 * x);
	} while (q > 0.27597 
		&& (q > 0.27846 || sqr(v) > -4.0 * log(u) * sqr(u)));
	return v / u;
}


int* histogram(double data[], int n_data, double low, double high, int n_bins) {
	int* bins = calloc(n_bins, sizeof(int));

	// TODO(eugenhotaj): We can speed this up by using binary search.
	double step_size = (high - low) / n_bins;
	for (int i = 0; i < n_data; i++) {
		for (int j = 0; j < n_bins; j++) {
			double boundary = low + (j+1) * step_size;
			if (data[i] < boundary || j == n_bins - 1) {
				bins[j] += 1;
				break;
			}
		}
	}
	return bins;
}


int main(void) {
	srand(time(NULL));

	int n = 10000, n_bins = 10;

	// Estimate the mean and variance.
	double sum = 0.0, ssum = 0.0;
	for (int i = 0; i < n; i++) {
		double sample = normal();
		sum += sample;
		ssum += sqr(sample);
	}	
	double mu = sum / n;
	double sig = (ssum / n) - sqr(mu);
	printf("mu: %f, sig: %f\n", mu, sig);


	// Createa histogram of samples.
	double samples[n];
	for (int i = 0; i < n; i++) {
		samples[i] = normal();
	}
	int* bins = histogram(samples, n, -3.0, 3.0, n_bins);

	for (int i = 0; i < n_bins; i++) {
		printf("%d ", bins[i]);
	}
	printf("\n");

    return 0;

    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
		"SDL Tutorial", 
		SDL_WINDOWPOS_UNDEFINED, 
		SDL_WINDOWPOS_UNDEFINED, 
		SCREEN_WIDTH, 
		SCREEN_HEIGHT, 
		SDL_WINDOW_SHOWN
	);
    SDL_Surface* screenSurface = SDL_GetWindowSurface(window);


	// Clear the screen.
	SDL_FillRect(screenSurface, NULL, SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));

	// Draw a rectangle.
	SDL_Rect rect = {.x = 50, .y = 50, .w = 100, .h = 100};
	SDL_FillRect(screenSurface, &rect, SDL_MapRGB(screenSurface->format, 0xFF, 0x00, 0x00));

	SDL_UpdateWindowSurface(window);

	//Hack to get window to stay up.
	SDL_Event e; 
	int quit = 0; 
	while(quit == 0) { 
		while(SDL_PollEvent(&e)){ 
			if(e.type == SDL_QUIT) {
				quit = 1; 
			} 
		}
	}

    SDL_DestroyWindow(window);
    SDL_Quit();
	return 0;
}

