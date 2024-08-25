#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <SDL2/SDL.h>

//Screen dimension constants.
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;

const int MAIN_RGBA[] = {0, 120, 255, 255};
const int ACCENT_RGBA[] = {0, 60, 120, 255};


double sqr(double x) {
	return x * x;
}

double max(double values[], int n_values) {
	double max_ = values[0];
	for (int i = 1; i < n_values; i++) {
		if (values[i] > max_) {
			max_ = values[i];
		}
	}
	return max_;
}

double min(double values[], int n_values) {
	double min_ = values[0];
	for (int i = 1; i < n_values; i++) {
		if (values[i] < min_) {
			min_= values[i];
		}
	}
	return min_;
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
double* histogram(double data[], int n_data, int n_bins) {
	double* bins = calloc(n_bins, sizeof(double));
	double x_min = min(data, n_data);
	double x_max = max(data, n_data);
	double step_size = (x_max - x_min) / n_bins;
	for (int i = 0; i < n_data; i++) {
		for (int j = 0; j < n_bins; j++) {
			double boundary = x_min + (j+1) * step_size;
			if (data[i] < boundary || j == n_bins - 1) {
				bins[j] += 1.0 / n_data;
				break;
			}
		}
	}
	return bins;
}

void draw_histogram(SDL_Renderer *renderer, double bins[], int n_bins, double y_min, double y_max) { 
    double width = (double) SCREEN_WIDTH / n_bins;
    int max_height = SCREEN_HEIGHT;

    // Draw histogram bars
    for (int i = 0; i < n_bins; i++) {
        int height = (bins[i] - y_min) / (y_max - y_min) * max_height;
        SDL_Rect rect = {
            .x = width * i,
            .y = SCREEN_HEIGHT - height,
            .w = width - 1,  // Leave 1 pixel gap between bars
            .h = height
        };
        SDL_SetRenderDrawColor(renderer, MAIN_RGBA[0], MAIN_RGBA[1], MAIN_RGBA[2], MAIN_RGBA[3]);
        SDL_RenderFillRect(renderer, &rect);
        
        // Draw outline
        SDL_SetRenderDrawColor(renderer, ACCENT_RGBA[0], ACCENT_RGBA[1], ACCENT_RGBA[2], ACCENT_RGBA[3]);
        SDL_RenderDrawRect(renderer, &rect);
    }
}


void draw_scatter_plot(SDL_Renderer *renderer, double x[], double y[], int n) {
    // Find min and max values for x and y.
    double x_min = min(x, n);
    double x_max = max(x, n);
    double y_min = min(y, n);
    double y_max = max(y, n);

    // Draw scatter plot points.
    SDL_SetRenderDrawColor(renderer, MAIN_RGBA[0], MAIN_RGBA[1], MAIN_RGBA[2], MAIN_RGBA[3]);
    for (int i = 0; i < n; i++) {
        int px = (x[i] - x_min) / (x_max - x_min) * SCREEN_WIDTH;
        int py = SCREEN_HEIGHT - (y[i] - y_min) / (y_max - y_min) * SCREEN_HEIGHT;
        
        SDL_Rect point = {px - 2, py - 2, 4, 4};
        SDL_RenderFillRect(renderer, &point);
    }
}


void clear_screen(SDL_Renderer *renderer) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
}

void show_and_wait(SDL_Renderer *renderer) {
	SDL_RenderPresent(renderer);
    SDL_Event e;
    int quit = 0;
    while (quit == 0) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = 1;
            }
        }
    }
	clear_screen(renderer);
}


int main(void) {
	srand(time(NULL));

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "Histogram",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	clear_screen(renderer);

	int n1 = 10000, n_bins = 50;

	// Samples from the standard normal.
	double samples[n1];
	for (int i = 0; i < n1; i++) {
		samples[i] = normal();
	}
	double* bins = histogram(samples, n1, n_bins);
	draw_histogram(renderer, bins, n_bins, 0, max(bins, n_bins) * 1.25);
	show_and_wait(renderer);


	// Samples from a mixture of two Gaussians.
	double mixture[n1];
	for (int i = 0; i < n1; i++) {
		if (uniform() < 0.3) {
			mixture[i] = normal() * 1.0 + 2.38;
		} else {
			mixture[i] = normal() * 1.2 - 1.3;
		}
	}
	double* mixture_bins = histogram(mixture, n1, n_bins);
	draw_histogram(renderer, mixture_bins, n_bins, 0, max(mixture_bins, n_bins) * 1.25);
	show_and_wait(renderer);

	// Draw a scatter plot.
	int n2 = 10;
	double x[n2], y[n2];
	for (int i = 0; i < n2; i++) {
		x[i] = normal();
		y[i] = normal();
	}
	draw_scatter_plot(renderer, x, y, n2);

	show_and_wait(renderer);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
	return 0;
}