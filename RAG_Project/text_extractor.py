import requests


def scrape_text_from_url():
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": "Eagle",
        "format": "json",
    }

    headers = {
        "User-Agent": "RAGProject/1.0 (student project)"
    }

    try:
        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch page. Status code: {response.status_code}")
            return ""

        data = response.json()
        pages = data["query"]["pages"]

        text = next(iter(pages.values()))["extract"]

        with open("Selected_Document.txt", "w", encoding="utf-8") as file:
            file.write(text)

        print("Success! Text saved to Selected_Document.txt")
        return text

    except Exception as error:
        print(f"Error: {error}")
        return ""


def main():
    scrape_text_from_url()


if __name__ == "__main__":
    main()