# tests/test_weather_tool.py

from unittest import TestCase
from tools.weather import get_geolocation, get_weather


class TestWeatherTool(TestCase):
    def test_get_geolocation1(self):
        """
        {"results":[{"id":4684888,"name":"Dallas","latitude":32.78306,"longitude":-96.80667,"elevation":128.0,"feature_code":"PPLA2","country_code":"US","admin1_id":4736286,"admin2_id":4684904,"timezone":"America/Chicago","population":1326087,"postcodes":["75201","75202","75203","75204","75205","75206","75207","75208","75209","75210","75211","75212","75214","75215","75216","75217","75218","75219","75220","75221","75222","75223","75224","75225","75226","75227","75228","75229","75230","75231","75232","75233","75234","75235","75236","75237","75238","75240","75241","75242","75243","75244","75246","75247","75248","75249","75250","75251","75252","75253","75254","75260","75261","75262","75263","75264","75265","75266","75267","75270","75275","75277","75283","75284","75285","75287","75301","75303","75312","75313","75315","75320","75326","75336","75339","75342","75354","75355","75356","75357","75359","75360","75367","75368","75370","75371","75372","75373","75374","75376","75378","75379","75380","75381","75382","75389","75390","75391","75392","75393","75394","75395","75397","75398"],"country_id":6252001,"country":"United States","admin1":"Texas","admin2":"Dallas"}],"generationtime_ms":0.5501509}
        """
        city = "Dallas"
        self.assertDictEqual(
            get_geolocation(city),
            {
                "name": "Dallas",
                "latitude": 32.78306,
                "longitude": -96.80667,
            },
        )

    def test_get_geolocation2(self):
        """
        {"results":[{"id":4719457,"name":"Plano","latitude":33.01984,"longitude":-96.69889,"elevation":203.0,"feature_code":"PPL","country_code":"US","admin1_id":4736286,"admin2_id":4682500,"timezone":"America/Chicago","population":283558,"postcodes":["75023","75024","75025","75026","75074","75075","75086","75093","75094"],"country_id":6252001,"country":"United States","admin1":"Texas","admin2":"Collin"}],"generationtime_ms":0.531435}
        """
        city = "Plano"
        self.assertDictEqual(
            get_geolocation(city),
            {
                "name": "Plano",
                "latitude": 33.01984,
                "longitude": -96.69889,
            },
        )

    def test_get_geolocation3(self):
        """
        {"results":[{"id":5391959,"name":"San Francisco","latitude":37.77493,"longitude":-122.41942,"elevation":16.0,"feature_code":"PPLA2","country_code":"US","admin1_id":5332921,"admin2_id":5391997,"timezone":"America/Los_Angeles","population":827526,"postcodes":["94102","94103","94104","94105","94107","94108","94109","94110","94111","94112","94114","94115","94116","94117","94118","94119","94120","94121","94122","94123","94124","94125","94126","94127","94129","94130","94131","94132","94133","94134","94137","94139","94140","94141","94142","94143","94144","94145","94146","94147","94151","94159","94160","94161","94163","94164","94172","94177","94188"],"country_id":6252001,"country":"United States","admin1":"California","admin2":"San Francisco County"}],"generationtime_ms":3.014803}
        """
        city = "San Francisco"
        self.assertDictEqual(
            get_geolocation(city),
            {
                "name": "San Francisco",
                "latitude": 37.77493,
                "longitude": -122.41942,
            },
        )
