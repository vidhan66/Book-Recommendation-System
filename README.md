# Book Recommendation System

## Overview

This project implements a Book Recommendation System using multiple recommendation techniques:

* Popularity-Based Recommendation

* Content-Based Filtering (CBF)

* Collaborative Filtering (CF)

* Hybrid Model (switching between CBF and CF)

The system suggests books to users based on different strategies, considering user preferences, book similarities, and overall popularity trends.

## Features

### 1. Popularity-Based Recommendation

* Recommends books based on their overall popularity (highest-rated books).

* Does not consider individual user preferences.

### 2. Content-Based Filtering (CBF)

* Suggests books similar to those a user has previously liked.

* Uses book metadata such as Book title and author to compute similarity.

* Active users: Users who have rated more than 100 books.

### 3. Collaborative Filtering (CF)

* Provides recommendations based on user behavior and preferences.

* Uses ratings from other users to suggest books.

* Defines:

  * **Active users:** Rated more than 100 books.

  * **Expert users:** Rated more than 300 books.

### 4. Hybrid Model

* Combines CBF and CF to leverage both content and user interactions.

* Provides a balanced recommendation strategy.

## Technologies Used

* Python

* Pandas, NumPy (Data manipulation and processing)

* Scikit-learn (Machine Learning models and preprocessing)

* Seaborn, Matplotlib (Data visualization)

* MinMaxScaler (Feature scaling)

* Cosine Similarity (Computing similarity between books)

* TF-IDF Vectorizer (Text-based similarity in CBF)

## Installation & Usage

### Prerequisites

Ensure you have the following installed:

```pip install pandas numpy scikit-learn seaborn matplotlib ```

## Contributing

We welcome contributions to enhance this project! To contribute:

**1. Fork the repository** – Click the ‘Fork’ button on GitHub.

**2. Create a new branch** – Use a meaningful name like feature-branch.

**3. Make your changes** – Ensure your code follows best practices and is well-documented.

**4. Commit your changes** – Write clear and descriptive commit messages.

**5. Push your changes** – Push your branch to your forked repository.

**6. Submit a pull request (PR)** – Explain what changes you made and why.

## Guidelines for Contributions:

* Issue Reporting: If you find a bug or have a feature request, open an issue on GitHub.

* Testing: Ensure your code runs correctly and does not break existing functionality.

* Documentation: Provide comments and update relevant documentation if needed.

We appreciate all contributions that help improve this project!

## License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute it. See the ```LICENSE``` file for details.

## Future Enhancements

* Implement Deep Learning-based recommendations.

* Improve hybrid model using dynamic weighting.

Developed by Vidhan Bansal. Feel free to contribute or reach out for collaborations!

