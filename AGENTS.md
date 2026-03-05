# Pull Request Messages

Pull request messages for this project should use the following format

```markdown
## Description

This section should contain a short description of what was included in the MR

## Testing

This section should contain instructions for how to validate the changes in the MR
```

Each pull request should have a title that uses conventional commit format.


## Update Code to Use Python Best Practices

This library was originally written in Python 2 and was also written with a
Mathematica or Maple usage pattern in mind. When you notice anti-patterns such
as `from applpy import *` or other aspects of the code, update them to use best practices.
Only make these updates incrementally in files we are updating to avoid unexpected changes.

## New Skills

Unless specifically requested, scope new skills created with the skill-creator skill to this repository. Only add new
skills to $HOME/.codex if it is specifically requested.
