# Wisdom — Quotes & Reflections

A [Hugo](https://gohugo.io/) site for collecting attributed quotes and personal reflections on them.

Live at [wisdom.accapehart.com](https://wisdom.accapehart.com/).

## Adding a Quote

```sh
hugo new quotes/attribution-topic.md
```

This scaffolds a post using the quotes archetype. Fill in the front matter fields:

| Field | Description |
|---|---|
| `quote` | The full quote text |
| `attribution` | Who said it |
| `attribution_confidence` | e.g. `confirmed`, `disputed`, `apocryphal`, `uncertain` |
| `link` | Source URL |
| `tags` | Themes (e.g. `['stoicism', 'integrity']`) |
| `draft` | Set to `false` when ready to publish |

Write your reflection in the body below the front matter.

## Local Development

```sh
hugo server -D   # -D includes drafts
```

## Building

```sh
hugo
```

Output goes to `public/`, which is gitignored.

## Theme

Uses [github-style](https://themes.gohugo.io/themes/github-style/) as a git submodule.
