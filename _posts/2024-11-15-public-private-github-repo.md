---
layout: blog
author: Ludwig Schneider
tags: [Software Engineering, GPU, GitHub]
---

# Syncing Public and Private GitHub Repositories for GPU Testing

For open-source projects like [AMSO](https://github.com/InnocentBug/AMSO), [Ptens](https://github.com/risi-kondor/ptens), and [PySSAGES](https://github.com/SSAGESLabs/PySAGES), automated testing is crucial not only for CPU platforms but also for GPUs, particularly Nvidia GPUs with CUDA code. While GitHub offers free runners for testing open-source projects (public repositories), there are limitations when it comes to GPU testing.

If you're new to setting up automated testing for your project, you can refer to my [introduction repository](https://github.com/InnocentBug/workflow-tutorial) or consult the comprehensive GitHub documentation.

## The Challenge: Automated Testing for GPU/CUDA Code

GitHub doesn't provide free GPU testing on their runners, which is understandable given the resource requirements. However, this presents a challenge for projects that rely heavily on GPU computations. To address this, we can repurpose old hardware and set up a [self-hosted GitHub Runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) with GPU capabilities.

While setting up a self-hosted runner is beyond the scope of this post, it's important to note that there are significant security considerations involved. One key precaution is to run tests only from private repositories on your self-hosted runner. This helps mitigate the risk of malicious actors attempting to compromise your runner through pull requests in public repositories.

## The Solution: Public-Private Repository Synchronization

To balance the need for GPU testing with security concerns and the open-source nature of our projects, we implement a dual-repository strategy:

1. A private repository for running GPU tests securely
2. A public repository for user access and visibility

This approach offers several benefits:

- Secure environment for GPU testing
- Ability to keep internal team discussions private
- Control over which versions are published publicly

While this post focuses on GPU testing, the public-private repository setup can be adapted for various scenarios where you need to maintain both private and public versions of a project.

In the following sections, we'll explore how to set up and synchronize these repositories effectively, ensuring that your open-source project remains accessible while allowing for secure GPU testing and internal development.

## Creating the Private Repository

Before we begin, ensure you have an existing public repository. For reference, you can view my public example repository [here](https://github.com/InnocentBug/runner-public).

Let's start by creating or using a local copy of the public repository:

```bash
git clone git@github.com:InnocentBug/runner-public.git
cd runner-public
```

Now, we'll create a private repository. There are two methods to achieve this:

### Method 1: Forking

1. Create a fork of the public repository in a new namespace.
2. Navigate to the settings tab of the forked repository on GitHub.
3. Change the visibility to `private`.
4. You'll receive a warning about detaching the fork from the public repository. This is expected and acceptable in this scenario.

### Method 2: New Repository (Recommended)

1. Create a new repository with a unique name directly on GitHub.
2. Set the visibility to `private`.

For this example, I've chosen the second method. You can view my private example repository [here](https://github.com/InnocentBug/runner-private).

After creating the empty private repository, we need to push the content from the public repository into it. Follow these steps:

1. Ensure you're in the local copy of the public repository.
2. Add the private repository as an additional remote target:

```bash
git remote add private git@github.com:YourUsername/your-private-repo.git
```

Replace `YourUsername` and `your-private-repo` with your GitHub username and the name of your private repository, respectively. This command assumes you're using SSH authentication with GitHub.

3. Push the desired branch (in this case, `main`) to the new private repository:

```bash
git push private main
```

From this point forward, most of your changes should be pushed directly to the private repository. Conduct your development, create pull requests, and perform other Git operations within the private repository.

**Note**: This setup allows you to maintain a public-facing repository while keeping sensitive or work-in-progress code in a private repository. It's particularly useful for projects that require GPU testing or have other specific needs that are best handled in a controlled, private environment. However, at this point the sync is manual, which is error-prone and laborious.

## Adding an Automated Test for the Private Repository

Now that we have our private repository set up, let's create a simple automated test that will run exclusively on our self-hosted runner. This test will serve as a starting point for more complex GPU-specific tests in the future.

### Creating the Workflow File

Create a new file named `.github/workflows/private_regular.yml` in your private repository with the following content:

```yml
name: Private Tests
permissions: read-all

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test-private:
    runs-on: ${{ secrets.SELF_HOSTED_RUNNER_NAME }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Run private test
        run: echo "Running Private Test on GPU"

      # Add more steps here for your GPU-specific tests
```

**Important:** Replace `${{ secrets.SELF_HOSTED_RUNNER_NAME }}` with the name of your self-hosted runner. For enhanced security, it's recommended to store the runner name as a secret in your GitHub repository settings.

### Adding the Workflow to the Repository

To add this workflow to your private repository, use the following Git commands:

```bash
git add .github/workflows/private_regular.yml
git commit -m "Add private GPU test workflow"
git push private main
```

Note that we're pushing directly to the private repository's main branch.

### Verifying the Workflow

After pushing the changes, navigate to the "Actions" tab in your private GitHub repository. You should see the "Private Tests" workflow listed. The workflow will be triggered automatically on pushes to the main branch or when pull requests targeting the main branch are created.

## Syncing to the Public Repository

To keep our private repository automatically in sync with the public repository, we'll set up another workflow. This workflow will push changes from the private repository to the public one, ensuring that the public version stays up-to-date with approved changes.

### Creating the Sync Workflow

Create a new file at `.github/workflows/public_sync.yml` with the following content:

```yml
name: Sync to Public Repository

on:
  workflow_dispatch:
  push:
    branches:
      - main
      - develop

jobs:
  sync-to-public:
    runs-on: ubuntu-latest
    steps:
      # Steps will be added here
```

This workflow is triggered in two scenarios:

1. Manually via `workflow_dispatch`, allowing you to sync on demand.
2. Automatically when changes are pushed to the `main` or `develop` branches.

**Note**: We're not using wildcards (`**`) for branch names to avoid syncing feature branches that might not be ready for public viewing.

### Setting Up Bot Permissions

To securely push changes to the public repository, we'll use a dedicated bot account. This approach offers better security management and easier revocation if needed.

1. Create a bot account. Meet mine: [InnocentBot](https://github.com/innocentBot).
2. Generate a dedicated SSH key for the bot:

```bash
ssh-keygen -t ed25519 -C "your_bot@example.com" -f ~/bot_rsa -N ""
```

This command creates an Ed25519 key pair without a passphrase, which is suitable for automated processes.

3. Add the public key to the bot's GitHub account:

   - Go to the bot's GitHub settings
   - Navigate to "SSH and GPG keys"
   - Click "New SSH key" and paste the contents of `~/bot_rsa.pub`

4. Add the bot as a collaborator to the **public** repository with appropriate permissions.

5. Store the private key as a secret in the **private** repository:
   - Go to the private repository's settings
   - Navigate to "Secrets and variables" > "Actions"
   - Create a new repository secret named `BOT_SSH_KEY`
   - Paste the contents of the private key file (`~/bot_rsa`)

### Configuring the Workflow

Now, let's add the necessary steps to our workflow:

```yml
jobs:
  sync-to-public:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout private repository
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Set up SSH agent
        uses: webfactory/ssh-agent@v0.7.0
        with:
          ssh-private-key: ${{ secrets.BOT_SSH_KEY }}

      - name: Configure Git
        run: |
          git config --global user.name 'InnocentBot'
          git config --global user.email 'your_bot@example.com'

      # Additional steps for syncing will be added here
```

Now we can access this key to make the pushes in the repository, specifically we are using the [webfactory/ssh-agent](https://github.com/marketplace/actions/webfactory-ssh-agent) for this purpose.
We also setup the git account to show the correct names for the bot in the `git log`.
This setup prepares the workflow to securely interact with the public repository using the bot's credentials.

## Preparing the Repository for Public Release

In this step, we'll configure the workflow to prepare the current branch for public release. This involves removing private workflows and activating public-only workflows.

### Removing Private Workflows

First, we'll remove workflows that should only run in the private repository:

```yml
- name: Remove private workflows
  run: |
    git rm .github/workflows/public_sync.yml
    git rm .github/workflows/private_regular.yml
```

This step removes the GPU testing workflow and the syncing workflow itself, as they should not be present in the public repository.

### Activating Public-Only Workflows

For workflows that should run only in the public repository (e.g., to utilize GitHub's free run time for public repositories), we'll move them from a special subdirectory into the main workflows directory:

```yml
- name: Activate public workflows
  run: |
    if [ -d ".github/workflows/only_public_runs" ]; then
      for file in .github/workflows/only_public_runs/*; do
        if [ -f "$file" ]; then
          git mv "$file" .github/workflows/
        fi
      done
    fi
```

This improved version checks if the directory exists and if the files are regular files before moving them, preventing potential errors.

**Important:** This method does not securely redact sensitive information. Any removed content remains in the repository history and will be visible in the public repository.

### Committing and Pushing Changes

After preparing the repository, we commit the changes and push them to the public repository:

```yml
- name: Commit changes
  run: git commit -am "Prepare for public release"

- name: Push to public repository
  run: |
    git remote add public git@github.com:InnocentBug/runner-public.git
    git push --force public ${{ github.ref_name }}
```

We use a force push to ensure the public repository exactly matches the private one. However, this approach has some considerations:

1. It overwrites any changes made directly to the public repository.
2. It may not be suitable if you expect contributions directly to the public repository.

### Alternative Approaches

To address these concerns, consider the following alternatives:

1. **Merge approach:** Pull from the public repository before committing changes. This preserves public contributions but may fail if there are merge conflicts:

```yml
- name: Sync with public repository
  run: |
    git remote add public git@github.com:InnocentBug/runner-public.git
    git fetch public
    git merge public/${{ github.ref_name }} --allow-unrelated-histories
```

2. **Staging branch approach:** Push to a staging branch for manual review:

```yml
- name: Push to staging branch
  run: |
    git remote add public git@github.com:InnocentBug/runner-public.git
    git push --force public ${{ github.ref_name }}:staging-${{ github.ref_name }}
```

This creates a `staging-{branch_name}` branch in the public repository, allowing for manual review and merging.

The staging approach offers more control and reduces the risk of accidental overwrites, making it ideal for projects where public contributions are expected or additional review is necessary before public release.

## Bidirectional Synchronization: Public to Private Repository

While the primary focus has been on syncing from private to public repositories, some projects may require bidirectional synchronization. This is particularly relevant when both external and internal collaborators are actively contributing to the project. However, for most smaller open-source projects, this level of complexity is often unnecessary, as external contributions, while welcome, are typically infrequent.

If you determine that bidirectional sync is essential for your project, you can adapt the previously described workflow. However, it's crucial to consider the following points:

### Security Considerations

- **SSH Key Management**: Placing a private SSH key in the public repository's secrets poses significant security risks. Ensure that this key is used exclusively for this sync operation and consider using separate bot accounts for each direction of synchronization.

- **Workflow Vigilance**: Pay extra attention to any changes in the public-to-private sync workflow and private workflows. These are potential vectors for malicious attacks and should be carefully monitored and reviewed.

### Preventing Circular Synchronization

To avoid an infinite loop of synchronization between repositories:

- Implement logic to exclude triggering the public-to-private sync if the most recent commit originated from a private-to-public push.
- Consider adding commit message tags or metadata to identify the source of each sync operation.

### Handling Concurrent Changes

The simple force-push approach doesn't account for simultaneous changes in both repositories, which can lead to data loss. To mitigate this:

- Implement a more sophisticated merging strategy that can handle concurrent changes.
- Consider using Git's `merge` command instead of force-pushing, and implement conflict resolution strategies.
- Be aware that "simultaneous" in this context can span several minutes due to workflow execution times.

### Recommended Approach for Smaller Projects

For most small to medium-sized open-source projects, manual synchronization from public to private repositories is often sufficient and safer. This approach allows for human oversight and reduces the risk of automated errors or security vulnerabilities.

### Scaling Considerations

As your project grows, you may need to explore more robust solutions:

- Consider using Git submodules or subtrees to manage public/private components separately.
- Investigate specialized tools or services designed for managing multi-repository projects.
- Implement more advanced CI/CD pipelines that can handle complex synchronization scenarios.

By carefully considering these factors, you can implement a bidirectional sync strategy that balances collaboration needs with security and data integrity concerns. Always prioritize the safety and stability of your project when implementing such advanced workflows.

## Summary

This guide has outlined a comprehensive approach to managing public and private GitHub repositories for open-source projects that require GPU testing. Here are the key takeaways:

1. **Dual Repository Setup**: We've established a system using both private and public repositories to balance security needs with open-source accessibility.

2. **Self-Hosted GPU Testing**: By utilizing self-hosted runners, we can perform GPU tests securely in a private environment.

3. **Automated Synchronization**: We've implemented a workflow to automatically sync changes from the private repository to the public one, ensuring that the open-source version stays up-to-date.

4. **Security Considerations**: Throughout the process, we've emphasized the importance of security, particularly in managing SSH keys and bot accounts.

5. **Workflow Customization**: The guide provides flexibility in how workflows are managed, including options for removing private workflows and activating public-only runs.

6. **Bidirectional Sync Considerations**: While not always necessary, we've discussed the potential need for and challenges of bidirectional synchronization between public and private repositories.

7. **Scalability**: For smaller projects, manual syncing from public to private may suffice, but we've acknowledged that larger projects might require more sophisticated solutions.

By following this approach, open-source projects can leverage the benefits of both private GPU testing and public collaboration, while maintaining security and code integrity. As with any development process, it's important to regularly review and adjust these practices to fit the evolving needs of your project and community.
