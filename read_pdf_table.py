import os
import re
import tabula  # tabula to extract tabular data
import pandas as pd

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTLine
from enum import Enum
from collections import defaultdict
from datetime import datetime


###########################################
# The following snippet was used to print the pdf elements of DKB-Kontoauszuege with the pdfminer
# in those pdf elements we can find lines matching the boarder of the table
# however long lines sometimes appear splitted, so we merge them in a meaningful way
# and thus obtain the table coordinates in the pdf which we pass to tabula which performs the actual parsing
# on the resulting pandas datafram we perform some additional clean up to have a nice result
#
#
# # snippet to print all elements
# from pdfminer.high_level import extract_pages
# for page_layout in extract_pages(file):
#     for element in page_layout:
#         print(element)
###########################################


class Point:

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def same_y(self, other: 'Point', tol=1e-7) -> bool:
        return abs(self.y - other.y) <= tol

    def same_x(self, other: 'Point', tol=1e-7) -> bool:
        return abs(self.x - other.x) <= tol


class StraightLine:
    def __init__(self, line: 'LTLine' = None, pts=None, linewidth=None) -> None:

        if line is not None:
            self.linewidth = line.linewidth

            assert len(line.pts) == 2, f'{line} is not straight line'

            self._p0 = Point(*line.pts[0])
            self._p1 = Point(*line.pts[1])
        else:
            self.linewidth = linewidth
            self._p0 = Point(*pts[0])
            self._p1 = Point(*pts[1])

        self.orientation = self.get_orientation()
        self._set_tblr()

        self._line = line

    def get_orientation(self):
        if self._p0.same_y(self._p1):
            return Orientation['H']
        elif self._p0.same_x(self._p1):
            return Orientation['V']
        else:
            return Orientation['UNDEFINED']

    def _set_tblr(self):
        # pdfminer coords(0, 0 in left lower corner)
        # tblr ==> top bottom left right ==> ymax, ymin, xmin, xmax
        if self.orientation == Orientation['H']:
            self._top = self._p0.y
            self._bottom = self._top
            self._left = min(self._p0.x, self._p1.x)
            self._right = max(self._p0.x, self._p1.x)
        else:
            self._top = max(self._p0.y, self._p1.y)
            self._bottom = min(self._p0.y, self._p1.y)
            self._left = self._p0.x
            self._right = self._left

    @property
    def top(self):
        # return top y position in pdfminer coords (0,0 in left lower corner)
        return self._top

    @property
    def bottom(self):
        # return bottom y position in pdfminer coords (0,0 in left lower corner)
        return self._bottom

    @property
    def left(self):
        # return left x position in pdfminer coords (0,0 in left lower corner)
        return self._left

    @property
    def right(self):
        # return right x position in pdfminer coords (0,0 in left lower corner)
        return self._right

    def get_const_position(self):
        if self.orientation == Orientation['H']:
            return self.top
        elif self.orientation == Orientation['V']:
            return self.left
        else:
            return None

    def to_interval(self):
        # ignoring const position only return changing coords
        if self.orientation == Orientation['H']:
            return [self.left, self.right]
        elif self.orientation == Orientation['V']:
            return [self.bottom, self.top]
        else:
            return []

    def length(self):
        if self.orientation == Orientation['H']:
            return self.right - self.left
        elif self.orientation == Orientation['V']:
            return self.top - self.bottom
        else:
            return None

    def __str__(self):
        if self.orientation == Orientation['H']:
            return f"{self.orientation} Straightline from x={self.left} to x={self.right} at y={self.top}"
        elif self.orientation == Orientation['V']:
            return f"{self.orientation} Straightline from y={self.bottom} to y={self.top} at x={self.right}"
        else:
            return self.__repr__()

    def __repr__(self):
        if self._line is not None:
            return f"StraightLine({self._line})"
        else:
            return f"StraightLine(pts=[{self._p0}, {self._p1}], linewidth={self.linewidth})"


class Orientation(Enum):
    H = 1  # --
    V = 2  # |
    UNDEFINED = 3


class StraightLineCollection:
    def __init__(self, lines: list[StraightLine], tol=2) -> None:
        self._lines = lines
        assert len(lines) > 0, 'Cant have empty collection'
        self.orientation = lines[0].orientation

        assert all(lines.orientation ==
                   self.orientation for lines in self._lines), 'Lines dont have matching orientations. Cant merge'
        assert self.orientation in [Orientation['H'], Orientation['V']
                                    ], f"Cant merge lines with orientation {self.orientation}"

        self.tol = tol  # tolerance threshold for merging

        self.groups = self.group()
        self.merge()

    def group(self):
        unique_pos = defaultdict(list)

        for l in self._lines:
            unique_pos[l.get_const_position()].append(l)
        return unique_pos

    def merge(self):
        if self.orientation == Orientation['H']:  # --
            self._lines = sorted(self._lines, key=lambda l: l.left)
            return
        elif self.orientation == Orientation['V']:  # |
            # for left and right vertical lines
            merged_group = dict()
            for const_pos, lines in self.groups.items():
                lines = sorted(lines, key=lambda l: l.bottom)
                # first: smallest bottom == lowest bottom since 0,0 is left bottom page corner
                # last is biggest bottom
                # do merge of lines:
                intervals = [lines[0].to_interval()]
                for l in lines[1:]:
                    if intervals[-1][0] - self.tol <= l.bottom <= intervals[-1][1] + self.tol:
                        # if l.bottom is in former interval with tolerance around than merge end points
                        intervals[-1][1] = max(l.top, intervals[-1][-1])
                    else:
                        intervals.append(l.to_interval())

                merged_lines = []
                for intvs in intervals:
                    line = StraightLine(pts=[(const_pos, intvs[0]), (const_pos, intvs[1])],
                                        linewidth=lines[0].linewidth)
                    merged_lines.append(line)
                merged_group[const_pos] = merged_lines

            self.groups = merged_group
            return

        else:
            raise RuntimeError('unsupported line orientation for merging')

    def __len__(self) -> int:
        return sum(len(v) for k, v in self.groups.items())

    def __str__(self) -> str:
        output = ''
        for k, v in self.groups.items():
            output += f"{k}:\n"
            for line in v:
                output += f"\t{str(line)}\n"
        return output


class Table:
    def __init__(self, path, hlines, vlines, page_idx, height, width) -> None:
        self.path = path
        self._hlines = hlines
        self._vlines = vlines
        self.page_idx = page_idx
        self.height = height
        self.width = width

        self.drop_page_headline()

        self.on_last_page = self.is_last_page()
        self._cleanup_lines()

        self.x1, self.y1, self.x2, self.y2 = self.get_table_coords()
        assert self._calc_area() > 0, f"invalid Table bbox"
        self.df = self.get_df()

    def _calc_area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def coord_trafo(self):
        # convert pdfminer coords (0,0 in left lower corner) to tabular coords (0,0 in left upper corner)
        # return in format fitting for area argument of "tabula.read_pdf" function: top,left,bottom,right == y1,x1,y2,x2

        top = self.height - max(self.y1, self.y2)
        bottom = self.height - min(self.y1, self.y2)

        left = min(self.x1, self.x2)
        right = max(self.x1, self.x2)
        return top, left, bottom, right

    def get_df(self):
        dfs = tabula.read_pdf(self.path,
                              pages=self.page_idx,
                              area=self.coord_trafo(),
                              silent=True, stream=True)
        assert len(dfs) == 1
        return dfs[0]

    def drop_page_headline(self):
        # every page as an hline below the DKB logo, we ignore this line:
        biggest_line_pos = max(self._hlines.groups.keys())
        del self._hlines.groups[biggest_line_pos]

    def _remove_lower_header_line(self):
        return self._remove_line(-2, use_hline=True)

    def _remove_line(self, sort_idx, use_hline=False, len_idx=0):
        # sort_idx: lowest idx is smallest value (most bottom if h, most left if v)
        lines = self._hlines.groups if use_hline else self._vlines.groups

        pos = sorted(lines)
        delete_key = pos[sort_idx]
        if len(lines[delete_key]) == 1:
            del lines[delete_key]
        else:
            # delete shorter one by default
            lenghts = sorted([l.length() for l in lines[delete_key]])
            lines[delete_key] = [l for l in lines[delete_key] if l.length() != lenghts[len_idx]]
        return

    def _cleanup_lines(self):
        # hlines
        self._remove_lower_header_line()
        if self.on_last_page:
            # delete bottom 2 hlines where alter/neuer kontostand is
            self._remove_line(0, use_hline=True)
            self._remove_line(0, use_hline=True)
            self._remove_line(1)  # middle vline, sort_idx refers to x-pos
            self._remove_line(0)  # smalles left vline

    def get_table_coords(self):
        lr = self._vlines.groups.keys()
        tb = self._hlines.groups.keys()

        x1, x2 = min(lr), max(lr)
        y1, y2 = min(tb), max(tb)

        return x1, y1, x2, y2

    def is_last_page(self):
        # last relevant page has table with alter/neuer kontostand
        # that makes 2 additional hlines and 2 more vertical lines
        # on other pages we only have 3 lines (2 from table header + 1 from bottom)

        if len(self._hlines) == 5 and len(self._vlines) == 4:
            return True

        return False

    def __str__(self) -> str:
        return str(self.df)


class Kontoauszug:
    def __init__(self, path) -> None:
        self.path = path

        self.tables = self.extract_tables()
        self.df = self.merge_tables()

        # TODO
        # do format checks for dfs
        # merge descriptions rows to rows with date
        self.df = self.merge_description_rows()
        self.df = self.enrich_df()

        # self.export(output_fmt='xlsx')

    def export(self, output_fmt='xlsx'):
        if output_fmt == 'xlsx':
            self.df.to_excel(self.path.replace('pdf', 'xlsx'))
        elif output_fmt == 'csv':
            self.df.to_csv(self.path.replace('pdf', 'csv'))
        elif output_fmt == 'json':
            self.df.to_json(self.path.replace('pdf', 'json'))
        else:
            raise NotImplementedError(
                (f"Output format {output_fmt} is not supported yet. "
                 "Look ad pandas.DataFrame.to_* functions to implement "
                 "your own export function"))

    def enrich_df(self):
        pat = "(\d\d\d\d)_(\d\d\d)"
        m = re.search(pat, os.path.basename(self.path))
        if m:
            year, number = [int(x) for x in m.groups()]
            # for number 1 and 13 we potentially have to concatenate next/previous year

            def append_year(row):
                attrs = ['Bu.Tag', 'Wert']
                for a in attrs:
                    if number == 1 and row[a].rstrip('.').split('.')[-1] == '12':
                        row[a] = datetime.strptime(f"{row[a]}{year-1}", "%d.%m.%Y")
                    elif number == 13 and row[a].rstrip('.').split('.')[-1] == '01':
                        row[a] = datetime.strptime(f"{row[a]}{year+1}", "%d.%m.%Y")
                    else:
                        row[a] = datetime.strptime(f"{row[a]}{year}", "%d.%m.%Y")
                return row

            return self.df.apply(append_year, axis=1)

        else:
            raise RuntimeError(f'Could not infere year and number from Kontoauszug filename {self.path}')

        return self.df

    def merge_description_rows(self):
        booking_day_idcs = self.df['Bu.Tag'].notna().to_numpy().nonzero()[0]
        records = []
        for start_idx, end_idx in zip(booking_day_idcs, booking_day_idcs[1:]):
            day_df = self.df.iloc[start_idx:end_idx]
            day_record = self._merge_descr_day(day_df)
            records.append(day_record)

        return pd.DataFrame.from_records(records)

    def _merge_descr_day(self, day_df):
        rec = dict()
        for k, v in day_df.iteritems():
            if k == 'Wir haben für Sie gebucht':  # description col
                rec[k] = ' '.join(v.tolist())
            else:
                rec[k] = v.iloc[0]
        return rec

    def merge_tables(self):
        return pd.concat([t.df for t in self.tables], ignore_index=True)

    def extract_tables(self):
        tables = []
        for page_idx, page_layout in enumerate(extract_pages(self.path)):
            hlines = []  # ---
            vlines = []  # |
            for elem in page_layout:
                if isinstance(elem, LTLine):
                    assert len(elem.pts) == 2, 'invalid line'

                    l = StraightLine(elem)
                    if l.orientation == Orientation['H']:
                        hlines.append(l)
                    elif l.orientation == Orientation['V']:
                        vlines.append(l)
            if len(hlines) > 0 and len(vlines) > 0:
                # ipdb.set_trace()
                hlines = StraightLineCollection(hlines)
                vlines = StraightLineCollection(vlines)
                # print(self.path + f'@ page {page_idx}')
                # print('hlines:')
                # print(hlines)
                # print('vlines:')
                # print(vlines)
                try:
                    tab = Table(path=self.path, hlines=hlines, vlines=vlines, page_idx=page_idx + 1,
                                height=page_layout.height, width=page_layout.width)
                    tables.append(tab)
                except AssertionError:
                    pass

        return tables

    def __str__(self) -> str:
        res = ''
        res += f"Kontoauszug of shape {self.df.shape}:\n"
        res += str(self.df)
        return res
