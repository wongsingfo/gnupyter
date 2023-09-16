# Plotter for CS Conference Papers

## Background & Motivation

We aim to create aesthetically pleasing graphs for our research papers, and while attempting to achieve this, we have encountered several challenges with gnuplot. These issues include:

1. **Limited Community Support:** Gnuplot suffers from a dwindling community of contributors and developers. This makes it challenging to swiftly find solutions to problems online.

2. **Lack of Essential Functionalities:** Gnuplot lacks certain fundamental features, such as the ability to simultaneously apply fill colors and markers in a histogram plot, which can be frustrating and unacceptable.

3. **Inadequate Data Processing Capabilities:** Gnuplot's limitations extend to its inability to perform certain data processing tasks, limiting its overall flexibility in data visualization.

As a solution to these challenges, we introduce our custom plotter, which is built upon JupyterLab, matplotlib, numpy, and pandas. This alternative not only enhances the visual appeal of our graphs but also simplifies the process and accelerates our graph creation workflow.

## Usage

### Docker Container

We provide a Dockerfile for creating a self-contained and portable environment for all functionalities. Simply run `docker build -t <tag> .` to build the image and `docker run -d -p 8888:8888 --name <name> <tag>` to run the container. The plotter environment can be accessed through `http://localhost:8888`

![page preview](image/page-preview.png)

### Local Installation

VSCode provides powerful Jupyter Notebook support, simply install [Jupyter Extension Package](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) and select the correct python3 kernel to launch the notebook.

## Template

We have developed a straightforward plotting template `template.ipynb` including commonly used graph types in conference papers. This template serves as a versatile tool for researchers aiming to visualize their data effectively. Please feel free to add more skeleton codes!