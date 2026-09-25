import type { APIRoute } from "astro";

const basePath = import.meta.env.BASE_URL.endsWith("/")
	? import.meta.env.BASE_URL
	: `${import.meta.env.BASE_URL}/`;
const sitemapUrl = new URL(`${basePath}sitemap-index.xml`, import.meta.env.SITE).href;

const robotsTxt = `
User-agent: *
Disallow: /_astro/

Sitemap: ${sitemapUrl}
`.trim();

export const GET: APIRoute = () => {
	return new Response(robotsTxt, {
		headers: {
			"Content-Type": "text/plain; charset=utf-8",
		},
	});
};
