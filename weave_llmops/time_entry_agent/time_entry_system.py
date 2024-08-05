import datetime

import weave


## Python set/get is weird but f it
@weave.op
class TimeEntry():
    date: datetime.datetime
    project: str
    code: int
    hours: int

    def __init__(self, date, project, code, hours):
        self.date = date
        self.project = project
        self.code = code
        self.hours = hours

@weave.op()
class TimeEntrySystem():
    ## organized as minisystem[entry.date][entry.project][entry.code]
    minisystem = {}

    def record_time(self, entry: TimeEntry) -> {}:
        if entry.date in self.minisystem:
            if entry.project in self.minisystem[entry.date]:
                if entry.code in self.minisystem[entry.date][entry.project]:
                    self.minisystem[entry.date][entry.project][entry.code] += entry.hours
                else:
                    self.minisystem[entry.date][entry.project][entry.code] = {entry.code: entry.hours}
            else:
                self.minisystem[entry.date][entry.project] = {entry.code: entry.hours}
        else:
            self.minisystem[entry.date] = {entry.project: {entry.code: entry.hours}}
        return self.minisystem[entry.date][entry.project][entry.code]
    def get_times_for_date(self, date: datetime) -> {}:
        return self.minisystem[date]

    def get_times_for_project(self, project: str):
        all_entries_for_project = {}
        for date in self.minisystem:
            if self.minisystem[date] == project:
                for project_code in self.minisystem[date][project]:
                    all_entries_for_project[project][date][project_code] = self.minisystem[date][project][project_code]
        return all_entries_for_project

    def get_times_for_code(self, code: int):
        all_entries_for_code = {}
        for date in self.minisystem:
            for project in self.minisystem[date]:
                if self.minisystem[date][project] == code:
                    all_entries_for_code[code][project][date] = self.minisystem[date][project][code]
        return all_entries_for_code

    def __str__(self):
        return str(self.minisystem)













