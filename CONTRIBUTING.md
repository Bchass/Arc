
# Contributing to Arc

Welcome to Arc! This guide outlines the steps and guidelines for contributing to Arc, including coding standards, testing procedures, and best practices.

### Getting Started

To contribute to Arc, you'll need to follow these steps:

1. Fork the Arc repository

2. Clone the Repository
   ```
   git clone https://github.com/your-username/arc.git
   ```

3. Create a new branch for your changes. It's recommended to name your branch in a descriptive manner that reflects the nature of your changes.
   ```
   git checkout -b descriptive-branch-name
   ```

### Coding Standards
  
- Follow PEP8 guidelines for code formatting.

### Testing

We use pytest for writing and running tests. Here's how to run the tests:
```
python -m pytest arc/matrices/tests
```

Make sure you have pytest installed:
```
pip install pytest
```

### Submitting Changes

Once you've made your changes and ensured they adhere to the coding standards and pass all tests, follow these steps to submit your contribution:

1. If there are conflicts in your branch due to changes in the main branch, rebase your branch onto the latest main branch to resolve conflicts.

2. Push your changes to your remote repository

3. Go to the GitHub page of your forked repository and create a pull request. Provide a clear description of the changes you've made.

4. Link the PR to the open issue. Ex: `closes #999` this will automatically close out the issue once the PR is merged. See more at: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests

