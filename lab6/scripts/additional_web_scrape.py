#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text


def get_data_by_th(soup, th_text):
    th = soup.find("th", string=th_text)
    if th and th.next_sibling:
        return th.next_sibling.get_text(strip=True)
    return None


def get_well_details(well_name=None, api_no=None):
    # Construct the first URL with parameters
    params = {
        "type": "wells",
    }
    if well_name:
        params["well_name"] = well_name
    if api_no:
        params["api_no"] = api_no

    results = {
        "api_no": api_no,
        "closest_city": None,
        "county": "",
        "latest_barrels_of_oil_produced": None,
        "latest_mcf_of_gas_produced": None,
        "latitude": 0.0,
        "link": "",
        "longitude": 0.0,
        "operator": "",
        "well_name": well_name,
        "well_status": None,
        "well_type": None,
    }

    response = requests.get('https://www.drillingedge.com/search', params=params)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the first href
        well_page_links = soup.find("table", class_="table wide-table interest_table").find("a")
        if well_page_links:
            well_page_link = well_page_links["href"]
            results["link"] = well_page_link
            response = requests.get(well_page_link)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                meta_info = soup.find("section", class_="meta_info")
                results["operator"] = meta_info.find_all("div")[2].find("span").text

                block_stats = meta_info.find_all("p", class_="block_stat")
                for stat in block_stats:
                    text = stat.get_text()
                    span_text = stat.find("span").text

                    text = text.replace(span_text, "").strip().split(" ")[:4]
                    text = " ".join(text).lower().replace(" ", "_")

                    results[f"latest_{text}"] = span_text.strip()

                well_table = soup.find("article", class_="well_table")
                if well_table:
                    results["well_status"] = get_data_by_th(well_table, "Well Status").strip()
                    results["well_type"] = get_data_by_th(well_table, "Well Type").strip()
                    results["closest_city"] = get_data_by_th(well_table, "Closest City").strip()
                    results["county"] = get_data_by_th(well_table, "County").strip()
                    results["well_name"] = get_data_by_th(well_table, "Well Name").strip()

            json_data_url = f"{well_page_link}?json"
            response = requests.get(json_data_url).json()
            results["latitude"] = float(response["data"][0]["lat"])
            results["longitude"] = float(response["data"][0]["lon"])
    return results


def extract_info(result):
    info = {
        "well_status": result['well_status'],
        "well_type": result['well_type'],
        "closest_city": result['closest_city'],
        "production_info": f"{result.get('latest_barrels_of_oil_produced', '0')} barrels of oil, "
                           f"{result.get('latest_mcf_of_gas_produced', '0')} mcf of gas"
    }
    return info

def update_database_with_scraped_info(engine, row_id, scraped_info):
    update_sql = """
    UPDATE oil_wells 
    SET well_status = :well_status,
        well_type = :well_type,
        closest_city = :closest_city,
        production_info = :production_info
    WHERE id = :id
    """
    with engine.begin() as conn:
        conn.execute(text(update_sql), {
            "well_status": scraped_info.get("well_status"),
            "well_type": scraped_info.get("well_type"),
            "closest_city": scraped_info.get("closest_city"),
            "production_info": scraped_info.get("production_info"),
            "id": row_id
        })

def main():
    db_url = "mysql+pymysql://phpmyadmin:Hyq010113!@localhost/oil_wells_db"
    engine = create_engine(db_url, echo=False)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, api, well_name FROM oil_wells"))
        rows = result.mappings().all()
    
    for row in rows:
        row_id = row["id"]
        api = row["api"]
        well_name = row["well_name"]
        if not api or not well_name:
            print(f"Record {row_id} missing API or well name, skipping.")
            continue
        print(f"Processing records {row_id}: API={api}, Well Name={well_name}")
        result = get_well_details(well_name, api)
        scraped_info = extract_info(result)
        if scraped_info:
            update_database_with_scraped_info(engine, row_id, scraped_info)
            print(f"Record {row_id} updated successfully, additional information: {scraped_info}")
        else:
            print(f"Failed to fetch additional information for record {row_id}.")
    
    print("All record additional information updated.")

if __name__ == "__main__":

    main()
