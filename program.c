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


// TODO(eugenhotaj): We can speed this up by using binary search.
double* histogram(double data[], int n_data, int n_bins, double low, double high) {
	double* bins = calloc(n_bins, sizeof(double));
	double step_size = (high - low) / n_bins;
	for (int i = 0; i < n_data; i++) {
		for (int j = 0; j < n_bins; j++) {
			double boundary = low + (j+1) * step_size;
			if (data[i] < boundary || j == n_bins - 1) {
				bins[j] += 1.0 / n_data;
				break;
			}
		}
	}
	return bins;
}

void draw_histogram(double bins[], int n_bins) {
	SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
		"Histogram", 
		SDL_WINDOWPOS_UNDEFINED, 
		SDL_WINDOWPOS_UNDEFINED, 
		SCREEN_WIDTH, 
		SCREEN_HEIGHT, 
		SDL_WINDOW_SHOWN
	);
	// The height of the histogram bins is drawn proportional to their probability mass. However, 
	// as the number of bins increases, their probability mass will tend to 0 and the overall 
	// histogram size will shrink. To decouple the histogram size from the number of bins we 
	// we rescale the bins so that the tallest bin always takes up 75% of the screen.
	double scale = 0.0;
	for (int i = 0; i < n_bins; i++) {
		if (bins[i] > scale) {
			scale = bins[i];
		}
	}
	scale *= 1.25;

	// Draw the histogram.
	SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
	SDL_RenderClear(renderer);
	double width = (double) SCREEN_WIDTH / n_bins;
	for (int i = 0; i < n_bins; i++) {
		int height = bins[i] * SCREEN_HEIGHT / scale;
		SDL_Rect rect = {
			.x = round(width * i), 
			.y = SCREEN_HEIGHT - height, 
			.w = i == 0 ? width : round(width * i / i), 
			.h = height
		};
		SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
		SDL_RenderFillRect(renderer, &rect);
	}
	SDL_RenderPresent(renderer);

	// Hack to get window to stay up.
	SDL_Event e; 
	int quit = 0; 
	while(quit == 0) { 
		while(SDL_PollEvent(&e)){ 
			if(e.type == SDL_QUIT) {
				quit = 1; 
			} 
		}
	}

    SDL_Quit();
}


int main(void) {
	srand(time(NULL));

	int n = 10000, n_bins = 50;
	
	// Samples from the standard normal.
	double samples[n];
	for (int i = 0; i < n; i++) {
		samples[i] = normal();
	}
	double* bins = histogram(samples, n, n_bins, -3.0, 3.0);
	draw_histogram(bins, n_bins);


	// Samples from a mixture of two Gaussians.
	double mixture[n];
	for (int i = 0; i < n; i++) {
		if (uniform() < 0.3) {
			mixture[i] = normal() * 1.0 + 2.38;
		} else {
			mixture[i] = normal() * 1.2 - 1.3;
		}
	}
	double* mixture_bins = histogram(mixture, n, n_bins, -5.0, 5.0);
	draw_histogram(mixture_bins, n_bins);

	return 0;
}