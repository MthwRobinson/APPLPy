# Pull Request Messages

Pull request messages for this project should use the following format

```markdown
## Description

This section should contain a short description of what was included in the MR

## Testing

This section should contain instructions for how to validate the changes in the MR
```

Each pull request should have a title that uses conventional commit format.

## Testing

If you modify rust code, run the following commands from the `Makefile` to confirm everything
is running as expected:

- `make cargo-lint`: runs `cargo clippy` and `cargo fmt --check` linting.
    If `cargo fmt --check` fails, run `make cargo-tidy` to format the code
- `make cargo-test`: runs the unit tests for the rust code

If you modify the python code, run the following commands to confirm everything is running
as expected:

- `make check`: runs `ruff` linting checks. Run `make tidy` to correct formatting differences
    if necessary
- `make rust-develop`: compiles the rust bindings to pick up any changes to the rust code
- `make test`: runs the python test suite

For Python targets, if you get an error saying `ModuleNotFound: applpy_rust`, that likely
means the rust bindings have not been built. You can fix this with `make rust-develop`.

## New Skills

Unless specifically requested, scope new skills created with the skill-creator skill to this repository. Only add new
skills to $HOME/.codex if it is specifically requested.
