# crawler_script.py
import asyncio
import sys
import logging
from crawl4ai import AsyncWebCrawler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def crawl_url(url):
    """Crawls the given URL and returns markdown content."""
    crawler = AsyncWebCrawler()
    markdown_content = None
    error_message = None
    try:
        logging.info(f"Starting crawl for: {url}")
        await crawler.start()
        result = await crawler.arun(url=url, magic=True)
        if result and result.markdown:
            markdown_content = result.markdown
            logging.info(f"Successfully crawled and extracted markdown from: {url}")
        else:
            error_message = f"Crawler finished for {url} but returned no markdown content."
            logging.warning(error_message)
    except Exception as e:
        error_message = f"An error occurred during crawling {url}: {e}"
        logging.exception(error_message) # Log the full traceback
    finally:
        await crawler.close()
        logging.info(f"Crawler closed for: {url}")
    return markdown_content, error_message

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python crawler_script.py <url>", file=sys.stderr)
        sys.exit(1)

    target_url = sys.argv[1]

    # Run the async function
    markdown, error = asyncio.run(crawl_url(target_url))

    if markdown:
        # Print the successful result (markdown) to standard output
        print(markdown)
        sys.exit(0) # Exit with success code
    else:
        # Print the error message to standard error
        print(f"Error crawling {target_url}: {error}", file=sys.stderr)
        sys.exit(1) # Exit with failure code