"""Time column read as real dates (or visit numbers, or labels) and the optional group column."""
from pathlib import Path

import numpy as np
import pytest

from longitudinal_impute.data import load_csv

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
NEXTGEN = Path(__file__).resolve().parents[1]
REPO = NEXTGEN.parent
COVID_SCHEMA = '[dataset]\nentity="location"\ntime="date"\ntotal="total_sequence"\ntargets=["v1","v2","v3","v4"]\n'
MICROBIOME_SCHEMA = ('[dataset]\nentity="subject"\ntime="visit"\ntargets=["Bacteroides","Prevotella",'
                     '"Faecalibacterium","Akkermansia","Bifidobacterium"]\n')
SMALL = '[dataset]\nentity="loc"\ntime="t"\ntotal="n"\ntargets=["a","b"]\n'


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def small(tmp_path, times, schema=SMALL, extra=''):
    body = ''.join(f'L{k},{t},10,1,2{extra}\n' for k, t in enumerate(times))
    header = 'loc,t,n,a,b' + (',grp' if extra else '') + '\n'
    return load_csv(write(tmp_path, 'data.csv', header + body), write(tmp_path, 'schema.toml', schema))


def test_iso_dates_become_days_since_the_first_date_and_week_index(tmp_path):
    data = load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 's.toml', COVID_SCHEMA))
    assert data.time_kind == 'date'
    # Avalon 2021-01-04 .. 2021-08-04, monthly; the other locations repeat the same dates.
    np.testing.assert_array_equal(data.days[:8], [0, 31, 59, 90, 120, 151, 181, 212])
    np.testing.assert_array_equal(data.days[8:16], data.days[:8])
    np.testing.assert_array_equal(data.week, data.days // 7)
    assert data.days.dtype == np.int64 and data.groups is None


def test_declared_time_format_is_used(tmp_path):
    data = small(tmp_path, ['04/01/2021', '18/01/2021'], SMALL + 'time_format="%d/%m/%Y"\n')
    assert data.time_kind == 'date'
    np.testing.assert_array_equal(data.days, [0, 14])
    np.testing.assert_array_equal(data.week, [0, 2])


@pytest.mark.parametrize('times,schema,message', [
    (['2021-01-04', '2021-02-30'], SMALL, r"CSV line 3: column 't' = '2021-02-30' is not a date in format '%Y-%m-%d'"),
    (['2021-01-04', '4 Jan'], SMALL, r"CSV line 3: column 't' = '4 Jan' is not a YYYY-MM-DD date"),
    (['2021-01-04', 'd1'], SMALL + 'time_format="%Y-%m-%d"\n', r"CSV line 3: column 't' = 'd1' is not a date"),
    (['2021/01/04'], SMALL + 'time_format="%Y-%m-%d"\n', r"CSV line 2: column 't' = '2021/01/04' is not a date"),
])
def test_bad_dates_are_rejected_with_line_column_and_format(tmp_path, times, schema, message):
    with pytest.raises(ValueError, match=message):
        small(tmp_path, times, schema)


def test_visit_numbers_are_kept_as_numbers(tmp_path):
    data = load_csv(FIXTURES / 'microbiome_small.csv', write(tmp_path, 's.toml', MICROBIOME_SCHEMA))
    assert data.time_kind == 'numeric' and data.week is None
    np.testing.assert_array_equal(data.days[:6], [1, 2, 3, 4, 5, 1])


def test_text_labels_and_absent_time_column(tmp_path):
    data = small(tmp_path, ['d2', 'd1', 'd10'])
    assert data.time_kind == 'label'
    np.testing.assert_array_equal(data.days, [2, 0, 1])        # rank among sorted labels
    data = load_csv(write(tmp_path, 'x.csv', 'loc,n,a,b\nL,10,1,2\n'),
                    write(tmp_path, 's.toml', '[dataset]\nentity="loc"\ntotal="n"\ntargets=["a","b"]\n'))
    assert data.days is None and data.week is None and data.time_kind is None and data.groups is None


def test_time_format_without_time_column_is_a_schema_error(tmp_path):
    with pytest.raises(ValueError, match='time_format needs a time column'):
        load_csv(write(tmp_path, 'x.csv', 'loc,n,a,b\nL,10,1,2\n'),
                 write(tmp_path, 's.toml', '[dataset]\nentity="loc"\ntotal="n"\ntargets=["a","b"]\n'
                                           'time_format="%Y"\n'))


def test_group_from_a_csv_column(tmp_path):
    data = small(tmp_path, ['1', '2'], SMALL + 'group_col="grp"\n', extra=',G1')
    assert data.groups == ('G1', 'G1')


def test_group_from_a_map_file_next_to_the_schema(tmp_path):
    write(tmp_path, 'map.csv', 'location,continent\nAvalon,North\nBrill,North\nCorin,South\nUnused,West\n')
    schema = COVID_SCHEMA + 'group_col="continent"\ngroup_map="map.csv"\n'
    data = load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 's.toml', schema))
    assert data.groups == ('North',) * 16 + ('South',) * 8


@pytest.mark.parametrize('map_text,schema_extra,message', [
    ('location,continent\nAvalon,North\n', 'group_map="map.csv"\n', r"no 'continent' for entities \['Brill', 'Corin'\]"),
    ('location,region\nAvalon,North\n', 'group_map="map.csv"\n', 'group map needs columns'),
    ('location,continent\nAvalon,North\nAvalon,South\n', 'group_map="map.csv"\n', "conflicting group for 'Avalon'"),
    ('', '', "group_col 'continent' is not a CSV column and the schema declares no group_map"),
])
def test_group_map_errors_are_explicit(tmp_path, map_text, schema_extra, message):
    write(tmp_path, 'map.csv', map_text)
    schema = COVID_SCHEMA + 'group_col="continent"\n' + schema_extra
    with pytest.raises(ValueError, match=message):
        load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 's.toml', schema))


def test_group_map_without_group_col_is_a_schema_error(tmp_path):
    with pytest.raises(ValueError, match='group_map needs group_col'):
        load_csv(FIXTURES / 'covid_small.csv', write(tmp_path, 's.toml', COVID_SCHEMA + 'group_map="m.csv"\n'))


@pytest.mark.skipif(not (REPO / 'data/covariants.csv').exists(), reason='full COVID table not in this checkout')
def test_shipped_covid_schema_maps_every_location_to_a_continent():
    data = load_csv(REPO / 'data/covariants.csv', NEXTGEN / 'configs/baselines_covariants.toml')
    assert data.time_kind == 'date' and data.days.min() == 0
    assert set(data.groups) == {'Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'}
