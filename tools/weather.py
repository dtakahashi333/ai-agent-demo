# tools/weather.py
import requests


def get_geolocation(city: str) -> dict:
    """
    Get the geo-location for a city using Open-Meteo API.
    """

    # Convert city name to coordinates
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    geocoding_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    geo_response = requests.get(
        geocoding_url,
        params=geocoding_params,
        timeout=10,
    )
    geo_response.raise_for_status()
    geo_data = geo_response.json()

    if not geo_data.get("results"):
        raise ValueError(f"City not found: {city}")

    location = geo_data["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    city_name = location["name"]

    return {
        "name": city_name,
        "latitude": latitude,
        "longitude": longitude,
    }


def get_weather(city: str) -> dict:
    """
    Get the current weather for a city using Open-Meteo API.
    """
    location = get_geolocation(city)

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
    }
    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=10,
    )
    weather_response.raise_for_status()
    weather_data = weather_response.json()

    current = weather_data["current"]

    return {
        "city": location["name"],
        "temperature": current["temperature_2m"],
        "temperature_unit": "°F",
        "humidity": current["relative_humidity_2m"],
        "wind_speed": current["wind_speed_10m"],
        "wind_speed_unit": "mph",
        "weather_code": current["weather_code"],
    }
