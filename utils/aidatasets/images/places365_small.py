import tarfile
import io
import datasets
from PIL import Image


class Places365Small(datasets.GeneratorBasedBuilder):
    """
    The Places365-Standard dataset (small version) for image classification.
    """

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="""
            Places365-Standard is a large-scale dataset for scene recognition, containing 1.8 million training images 
            and 36,500 validation images over 365 scene categories.
            """,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="http://places2.csail.mit.edu/",
            license="MIT License",
            citation="""@article{zhou2017places,
                         title={Places: A 10 million Image Database for Scene Recognition},
                         author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
                         year={2017}}
            """,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download("http://data.csail.mit.edu/places/places365/train_256_places365standard.tar")
        val_path = dl_manager.download("http://data.csail.mit.edu/places/places365/val_256.tar")
        devkit_path = dl_manager.download("http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"archive_path": train_path, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"archive_path": val_path, "split": "val", "devkit_path": devkit_path},
            ),
        ]

    def _generate_examples(self, archive_path, split, devkit_path=None):
        with tarfile.open(archive_path, "r") as tar:
            if split == "train":
                for member in tar.getmembers():
                    if member.isfile():
                        # Extract image as file-like object
                        file_like = tar.extractfile(member)
                        if file_like:
                            # Convert to PIL.Image
                            image = Image.open(io.BytesIO(file_like.read()))
                            label = self.extract_train_class(member.name)
                            yield member.name, {"image": image, "label": label}

            elif split == "val":
                folder = "val_256/"

                # Extract places365_val.txt from devkit
                label_mapping = {}
                with tarfile.open(devkit_path, "r") as devkit_tar:
                    val_label_file = devkit_tar.extractfile("places365_val.txt")
                    for line in val_label_file:
                        filename, class_idx = line.decode("utf-8").strip().split()
                        label_mapping[filename] = int(class_idx)

                # Process validation images
                for member in tar.getmembers():
                    if member.isfile():
                        filename = member.name[len(folder):]  # Remove "val_256/" prefix
                        if filename in label_mapping:
                            # Extract image as file-like object
                            file_like = tar.extractfile(member)
                            if file_like:
                                # Convert to PIL.Image
                                image = Image.open(io.BytesIO(file_like.read()))
                                class_idx = label_mapping[filename]
                                label_name = self._labels()[class_idx]  # Map class index to label name
                                yield member.name, {"image": image, "label": label_name}

    @staticmethod
    def extract_train_class(input_string):
        # Find the first '/' and trim everything before it
        first_slash_index = input_string.find('/')
        trimmed_start = input_string[first_slash_index:]

        # Remove everything after the last '/'
        last_slash_index = trimmed_start.rfind('/')
        result = trimmed_start[:last_slash_index]

        return result

    @staticmethod
    def _labels():
        return ['/a/airfield', '/a/airplane_cabin', '/a/airport_terminal', '/a/alcove', '/a/alley', '/a/amphitheater',
                '/a/amusement_arcade', '/a/amusement_park', '/a/apartment_building/outdoor', '/a/aquarium',
                '/a/aqueduct', '/a/arcade', '/a/arch', '/a/archaelogical_excavation', '/a/archive', '/a/arena/hockey',
                '/a/arena/performance', '/a/arena/rodeo', '/a/army_base', '/a/art_gallery', '/a/art_school',
                '/a/art_studio', '/a/artists_loft', '/a/assembly_line', '/a/athletic_field/outdoor', '/a/atrium/public',
                '/a/attic', '/a/auditorium', '/a/auto_factory', '/a/auto_showroom', '/b/badlands', '/b/bakery/shop',
                '/b/balcony/exterior', '/b/balcony/interior', '/b/ball_pit', '/b/ballroom', '/b/bamboo_forest',
                '/b/bank_vault', '/b/banquet_hall', '/b/bar', '/b/barn', '/b/barndoor', '/b/baseball_field',
                '/b/basement', '/b/basketball_court/indoor', '/b/bathroom', '/b/bazaar/indoor', '/b/bazaar/outdoor',
                '/b/beach', '/b/beach_house', '/b/beauty_salon', '/b/bedchamber', '/b/bedroom', '/b/beer_garden',
                '/b/beer_hall', '/b/berth', '/b/biology_laboratory', '/b/boardwalk', '/b/boat_deck', '/b/boathouse',
                '/b/bookstore', '/b/booth/indoor', '/b/botanical_garden', '/b/bow_window/indoor', '/b/bowling_alley',
                '/b/boxing_ring', '/b/bridge', '/b/building_facade', '/b/bullring', '/b/burial_chamber',
                '/b/bus_interior', '/b/bus_station/indoor', '/b/butchers_shop', '/b/butte', '/c/cabin/outdoor',
                '/c/cafeteria', '/c/campsite', '/c/campus', '/c/canal/natural', '/c/canal/urban', '/c/candy_store',
                '/c/canyon', '/c/car_interior', '/c/carrousel', '/c/castle', '/c/catacomb', '/c/cemetery', '/c/chalet',
                '/c/chemistry_lab', '/c/childs_room', '/c/church/indoor', '/c/church/outdoor', '/c/classroom',
                '/c/clean_room', '/c/cliff', '/c/closet', '/c/clothing_store', '/c/coast', '/c/cockpit',
                '/c/coffee_shop', '/c/computer_room', '/c/conference_center', '/c/conference_room',
                '/c/construction_site', '/c/corn_field', '/c/corral', '/c/corridor', '/c/cottage', '/c/courthouse',
                '/c/courtyard', '/c/creek', '/c/crevasse', '/c/crosswalk', '/d/dam', '/d/delicatessen',
                '/d/department_store', '/d/desert/sand', '/d/desert/vegetation', '/d/desert_road', '/d/diner/outdoor',
                '/d/dining_hall', '/d/dining_room', '/d/discotheque', '/d/doorway/outdoor', '/d/dorm_room',
                '/d/downtown', '/d/dressing_room', '/d/driveway', '/d/drugstore', '/e/elevator/door',
                '/e/elevator_lobby', '/e/elevator_shaft', '/e/embassy', '/e/engine_room', '/e/entrance_hall',
                '/e/escalator/indoor', '/e/excavation', '/f/fabric_store', '/f/farm', '/f/fastfood_restaurant',
                '/f/field/cultivated', '/f/field/wild', '/f/field_road', '/f/fire_escape', '/f/fire_station',
                '/f/fishpond', '/f/flea_market/indoor', '/f/florist_shop/indoor', '/f/food_court', '/f/football_field',
                '/f/forest/broadleaf', '/f/forest_path', '/f/forest_road', '/f/formal_garden', '/f/fountain',
                '/g/galley', '/g/garage/indoor', '/g/garage/outdoor', '/g/gas_station', '/g/gazebo/exterior',
                '/g/general_store/indoor', '/g/general_store/outdoor', '/g/gift_shop', '/g/glacier', '/g/golf_course',
                '/g/greenhouse/indoor', '/g/greenhouse/outdoor', '/g/grotto', '/g/gymnasium/indoor', '/h/hangar/indoor',
                '/h/hangar/outdoor', '/h/harbor', '/h/hardware_store', '/h/hayfield', '/h/heliport', '/h/highway',
                '/h/home_office', '/h/home_theater', '/h/hospital', '/h/hospital_room', '/h/hot_spring',
                '/h/hotel/outdoor', '/h/hotel_room', '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_cream_parlor',
                '/i/ice_floe', '/i/ice_shelf', '/i/ice_skating_rink/indoor', '/i/ice_skating_rink/outdoor',
                '/i/iceberg', '/i/igloo', '/i/industrial_area', '/i/inn/outdoor', '/i/islet', '/j/jacuzzi/indoor',
                '/j/jail_cell', '/j/japanese_garden', '/j/jewelry_shop', '/j/junkyard', '/k/kasbah',
                '/k/kennel/outdoor', '/k/kindergarden_classroom', '/k/kitchen', '/l/lagoon', '/l/lake/natural',
                '/l/landfill', '/l/landing_deck', '/l/laundromat', '/l/lawn', '/l/lecture_room',
                '/l/legislative_chamber', '/l/library/indoor', '/l/library/outdoor', '/l/lighthouse', '/l/living_room',
                '/l/loading_dock', '/l/lobby', '/l/lock_chamber', '/l/locker_room', '/m/mansion',
                '/m/manufactured_home', '/m/market/indoor', '/m/market/outdoor', '/m/marsh', '/m/martial_arts_gym',
                '/m/mausoleum', '/m/medina', '/m/mezzanine', '/m/moat/water', '/m/mosque/outdoor', '/m/motel',
                '/m/mountain', '/m/mountain_path', '/m/mountain_snowy', '/m/movie_theater/indoor', '/m/museum/indoor',
                '/m/museum/outdoor', '/m/music_studio', '/n/natural_history_museum', '/n/nursery', '/n/nursing_home',
                '/o/oast_house', '/o/ocean', '/o/office', '/o/office_building', '/o/office_cubicles', '/o/oilrig',
                '/o/operating_room', '/o/orchard', '/o/orchestra_pit', '/p/pagoda', '/p/palace', '/p/pantry', '/p/park',
                '/p/parking_garage/indoor', '/p/parking_garage/outdoor', '/p/parking_lot', '/p/pasture', '/p/patio',
                '/p/pavilion', '/p/pet_shop', '/p/pharmacy', '/p/phone_booth', '/p/physics_laboratory',
                '/p/picnic_area', '/p/pier', '/p/pizzeria', '/p/playground', '/p/playroom', '/p/plaza', '/p/pond',
                '/p/porch', '/p/promenade', '/p/pub/indoor', '/r/racecourse', '/r/raceway', '/r/raft',
                '/r/railroad_track', '/r/rainforest', '/r/reception', '/r/recreation_room', '/r/repair_shop',
                '/r/residential_neighborhood', '/r/restaurant', '/r/restaurant_kitchen', '/r/restaurant_patio',
                '/r/rice_paddy', '/r/river', '/r/rock_arch', '/r/roof_garden', '/r/rope_bridge', '/r/ruin', '/r/runway',
                '/s/sandbox', '/s/sauna', '/s/schoolhouse', '/s/science_museum', '/s/server_room', '/s/shed',
                '/s/shoe_shop', '/s/shopfront', '/s/shopping_mall/indoor', '/s/shower', '/s/ski_resort', '/s/ski_slope',
                '/s/sky', '/s/skyscraper', '/s/slum', '/s/snowfield', '/s/soccer_field', '/s/stable',
                '/s/stadium/baseball', '/s/stadium/football', '/s/stadium/soccer', '/s/stage/indoor',
                '/s/stage/outdoor', '/s/staircase', '/s/storage_room', '/s/street', '/s/subway_station/platform',
                '/s/supermarket', '/s/sushi_bar', '/s/swamp', '/s/swimming_hole', '/s/swimming_pool/indoor',
                '/s/swimming_pool/outdoor', '/s/synagogue/outdoor', '/t/television_room', '/t/television_studio',
                '/t/temple/asia', '/t/throne_room', '/t/ticket_booth', '/t/topiary_garden', '/t/tower', '/t/toyshop',
                '/t/train_interior', '/t/train_station/platform', '/t/tree_farm', '/t/tree_house', '/t/trench',
                '/t/tundra', '/u/underwater/ocean_deep', '/u/utility_room', '/v/valley', '/v/vegetable_garden',
                '/v/veterinarians_office', '/v/viaduct', '/v/village', '/v/vineyard', '/v/volcano',
                '/v/volleyball_court/outdoor', '/w/waiting_room', '/w/water_park', '/w/water_tower', '/w/waterfall',
                '/w/watering_hole', '/w/wave', '/w/wet_bar', '/w/wheat_field', '/w/wind_farm', '/w/windmill', '/y/yard',
                '/y/youth_hostel', '/z/zen_garden']

