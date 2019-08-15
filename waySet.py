from shapely.geometry import LineString, MultiLineString
import overpy
import time

import matplotlib.pyplot as plt
import json

import route


class ObjectCache:

    def __init__(self):
        self.query_cache = None

    def get_from_cache(self, query):
        try:
            if self.query_cache is None:
                with open("query_cache.json", 'r') as f:
                    self.query_cache = json.load(f)

            return self.query_cache[query]
        except Exception:
            return None

    def store_in_cache(self, query, data):
        if self.query_cache is None:
            self.query_cache = {}
        self.query_cache[query] = data

        with open("query_cache.json", 'w') as outfile:
            json.dump(self.query_cache, outfile, indent=4, sort_keys=True)


class WaySet:

    def __init__(self, lines):
        self.multiline = MultiLineString(lines)

    ways_cache = ObjectCache()

    @staticmethod
    def download_all_ways(sector, onlyTram=False):
        bbox = "%f,%f,%f,%f" % sector
        ext = 0.01
        ext_bbox = "%f,%f,%f,%f" % (sector[0] - ext, sector[1] - ext, sector[2] + ext, sector[3] + ext)

        if onlyTram:
            query = '[out:json];(node({{ext_bbox}});way["railway"="tram"]({{bbox}}););out;'
        else:
            query = '[out:json];(way["highway"]({{bbox}});node({{ext_bbox}}););out;'

        query = query.replace("{{bbox}}", bbox)
        query = query.replace("{{ext_bbox}}", ext_bbox)

        ways = WaySet.ways_cache.get_from_cache(query)
        if ways is None:
            api = overpy.Overpass()
            try:
                result = api.query(query)
            except:
                time.sleep(20)
                result = api.query(query)

            ways = []
            for w in result.ways:
                try:
                    nodes = w.get_nodes(resolve_missing=False)
                except:
                    nodes = w.get_nodes(resolve_missing=True)
                nodes = [[float(n.lon), float(n.lat)] for n in nodes]
                ways += [nodes]

            WaySet.ways_cache.store_in_cache(query, ways)

        lines = [LineString(c) for c in ways]

        return WaySet(lines)

    def snap_point_to_lines(self, point, next_point, priority_line_index):
        distances = [line.distance(point) for line in self.multiline]

        if priority_line_index > -1:
            distances[priority_line_index] /= 2

        min_distance = min(distances)
        i_dist = distances.index(min_distance)

        projection = self.multiline[i_dist].project(point)
        next_proj = 1 if next_point is None else self.multiline[i_dist].project(next_point)
        new_point = self.multiline[i_dist].interpolate(projection)
        new_next_point = self.multiline[i_dist].interpolate(next_proj)

        heading = route.Route.get_heading(new_point.y, new_point.x, new_next_point.y, new_next_point.x)

        return new_point, i_dist, heading

    def snap_point_to_multilines(self, point):
        return self.multiline.interpolate(self.multiline.project(point))

    def plot(self):
        for l in self.multiline:
            lon, lat = l.xy
            plt.plot(lon, lat)

    def get_closest_way(self, point):
        distances = [line.distance(point) for line in self.multiline]
        min_distance = min(distances)
        i_min_dist = distances.index(min_distance)
        return self.multiline[i_min_dist]

    def snap_points(self, points):
        last_way = None
        snapped_points = []
        last_distance = None
        for point_index in range(len(points)):

            point = points[point_index]

            if last_way is not None:
                projection = last_way.project(point)
                distance = last_way.distance(point)
                if 0 <= projection <= 1 and distance < 1.1 * last_distance:
                    snapped_point = last_way.interpolate(projection)
                    last_distance = distance
                    snapped_points += [snapped_point]
                    continue

            last_way = self.get_closest_way(point)
            projection = last_way.project(point)
            last_distance = last_way.distance(point)
            snapped_point = last_way.interpolate(projection)

            snapped_points += [snapped_point]

        headings = []
        for i, p in enumerate(snapped_points):
            if i < len(snapped_points)-1:
                next_point = snapped_points[i+1]
                headings.append(route.Route.get_heading(p.y, p.x, next_point.y, next_point.x))
        headings.append(headings[-1])

        return route.Route.from_points(snapped_points, headings)
