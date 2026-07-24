# Ziheng Zhao — Academic Website

Personal academic website and research notebook built with the
[al-folio](https://github.com/alshedivat/al-folio) Jekyll theme.

## Writing a blog post

1. Copy `_drafts/post-draft.md` to `_posts/YYYY-MM-DD-title.md`.
2. Update the title, description, tags, categories, and date.
3. Write the post in Markdown.
4. Commit and push to `master`; GitHub Actions will build and publish the site.

## Main content

- Homepage: `_pages/about.md`
- Blog index: `_pages/blog.md`
- Publications: `_pages/publications.md`
- Publication data: `_bibliography/papers.bib`
- CV data: `_data/cv.yml`
- Downloadable CV: `assets/pdf/ziheng-zhao-cv.pdf`
- Social links: `_data/socials.yml`

## Local preview

The recommended local workflow is Docker:

```bash
docker compose up
```

The site will be available at `http://localhost:8080`.
