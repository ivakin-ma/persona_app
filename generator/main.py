import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import database
from generator import generate_persona, generate_many
from ai_analyzer import analyze_personas


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Генератор и анализ архетипов пользователей")
        self.geometry("1480x760")
        database.init_db()
        self.create_widgets()
        self.refresh()

    def create_widgets(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Добавить вручную", command=self.open_manual_form).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Сгенерировать 1", command=self.generate_one).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Массовая генерация", command=self.bulk_generate).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Нейроанализ и советы", command=self.run_ai_analysis).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Карточка пользователя", command=self.show_selected_details).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Очистить базу данных", command=self.clear_database).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Обновить", command=self.refresh).pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(top, text="")
        self.status.pack(side=tk.RIGHT)

        columns = (
            "id", "name", "gender", "age", "occupation",
            "occupation_group", "device", "device_type", "pain_points"
        )
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=17)
        headers = {
            "id": "ID",
            "name": "Имя",
            "gender": "Пол",
            "age": "Возраст",
            "occupation": "Профессия",
            "occupation_group": "Группа профессий",
            "device": "Устройство",
            "device_type": "Тип устройства",
            "pain_points": "Боль / барьер",
        }
        widths = {
            "id": 45, "name": 110, "gender": 110, "age": 70,
            "occupation": 160, "occupation_group": 160, "device": 210, "device_type": 140,
            "pain_points": 360,
        }
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=widths.get(col, 120), anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        bottom = ttk.LabelFrame(self, text="Статистика, карточка и рекомендации", padding=10)
        bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.output = tk.Text(bottom, height=15, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True)

    def refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = database.list_personas()
        for row in rows:
            self.tree.insert("", tk.END, values=(
                row["id"], row["name"], row["gender"], row["age"],
                row["occupation"], row["occupation_group"], row["device"], row["device_type"],
                row["pain_points_text"]
            ))
        self.status.config(text=f"Записей: {database.count_personas()}")

    def generate_one(self):
        try:
            persona = generate_persona()
        except Exception as exc:
            messagebox.showerror("Ошибка генерации Qwen", str(exc))
            return
        database.add_persona(persona)
        self.refresh()

    def bulk_generate(self):
        count = simpledialog.askinteger("Массовая генерация", "Сколько архетипов создать?", minvalue=1, maxvalue=100000)
        if not count:
            return
        try:
            personas = generate_many(count)
        except Exception as exc:
            messagebox.showerror("Ошибка генерации Qwen", str(exc))
            return
        database.bulk_add(personas)
        self.refresh()
        messagebox.showinfo("Готово", f"Создано записей: {count}")

    def clear_database(self):
        if messagebox.askyesno("Очистить базу данных", "Удалить все архетипы из базы данных?"):
            database.clear_database()
            self.output.delete("1.0", tk.END)
            self.refresh()

    def open_manual_form(self):
        win = tk.Toplevel(self)
        win.title("Ручной ввод архетипа")
        win.geometry("620x620")
        win.transient(self)
        win.grab_set()

        fields = [
            ("name", "Имя"),
            ("gender", "Пол (м, ж, муж, жен, мужской, женский; иначе определит ИИ при анализе)"),
            ("age", "Возраст"),
            ("occupation", "Профессия"),
            ("device", "Устройство / модель"),
            ("context", "Контекст использования"),
            ("goal", "Цель пользователя"),
            ("behavior", "Поведение пользователя"),
            ("pain", "Боль / барьер пользователя"),
        ]
        entries = {}
        form = ttk.Frame(win, padding=12)
        form.pack(fill=tk.BOTH, expand=True)

        for i, (key, label) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(form, width=58)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=5)
            entries[key] = entry

        form.columnconfigure(1, weight=1)

        def save():
            persona = {
                "name": entries["name"].get().strip(),
                "gender": entries["gender"].get().strip(),
                "age": entries["age"].get().strip(),
                "occupation": entries["occupation"].get().strip(),
                "device": entries["device"].get().strip(),
                "context": entries["context"].get().strip(),
                "avatar_path": "",
                "goals": [{"type": "end", "description": entries["goal"].get().strip()}] if entries["goal"].get().strip() else [],
                "behaviors": [{"pattern": entries["behavior"].get().strip(), "description": entries["behavior"].get().strip()}] if entries["behavior"].get().strip() else [],
                "pain_points": [entries["pain"].get().strip()] if entries["pain"].get().strip() else [],
            }
            if not persona["name"]:
                messagebox.showwarning("Ошибка", "Введите имя пользователя.", parent=win)
                return
            database.add_persona(persona)
            self.refresh()
            win.destroy()

        ttk.Button(form, text="Сохранить", command=save).grid(row=len(fields), column=1, sticky=tk.E, pady=16)

    def _selected_persona_id(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Не выбрано", "Выберите архетип в таблице.")
            return None
        values = self.tree.item(selected[0], "values")
        return int(values[0])

    def show_selected_details(self):
        persona_id = self._selected_persona_id()
        if persona_id is None:
            return
        details = database.get_persona_details(persona_id)
        if not details:
            return
        p = details["persona"]
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Карточка пользователя\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, f"Имя: {p['name']}\n")
        self.output.insert(tk.END, f"Пол: {p['gender']}\n")
        self.output.insert(tk.END, f"Возраст: {p['age']}\n")
        self.output.insert(tk.END, f"Профессия: {p['occupation']}\n")
        self.output.insert(tk.END, f"Группа профессий: {p['occupation_group']}\n")
        self.output.insert(tk.END, f"Устройство: {p['device']}\n")
        self.output.insert(tk.END, f"Тип устройства: {p['device_type']}\n")
        self.output.insert(tk.END, "\n")

        self.output.insert(tk.END, "Цели:\n")
        for goal in details["goals"]:
            self.output.insert(tk.END, f"- [{goal['goal_type']}] {goal['description']}\n")
        self.output.insert(tk.END, "\nПоведение:\n")
        for behavior in details["behaviors"]:
            text = behavior["description"] or behavior["pattern"]
            self.output.insert(tk.END, f"- {text}\n")
        self.output.insert(tk.END, "\nБоли / барьеры:\n")
        for pain in details["pain_points"]:
            self.output.insert(tk.END, f"- {pain['description']}\n")

    def run_ai_analysis(self):
        rows = database.list_personas_for_analysis()
        if not rows:
            messagebox.showwarning("Нет данных", "Сначала добавьте или сгенерируйте архетипы.")
            return

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Выполняется анализ...\n")
        self.update_idletasks()

        def save_fields(pid, gender, device_type, occupation_group):
            database.update_analysis_fields(pid, gender, device_type, occupation_group)

        result = analyze_personas(rows, save_callback=save_fields)
        self.refresh()

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"Всего архетипов: {result['total']}\n")
        self.output.insert(tk.END, f"Модель: {result['model']}\n")
        self.output.insert(tk.END, f"Устройство: {result['device'] or 'не определено'}\n")
        self.output.insert(tk.END, f"Нейросеть загружена: {'да' if result['model_loaded'] else 'нет, используется резервная логика'}\n")
        if result["model_error"]:
            self.output.insert(tk.END, f"Ошибка модели: {result['model_error']}\n")
        self.output.insert(tk.END, "\n")

        sections = [
            ("Статистика по полу", result["gender_stats"]),
            ("Статистика по типам устройств", result["device_stats"]),
            ("Схожие профессии по группам", result["occupation_stats"]),
            ("Статистика по болям / барьерам", result["pain_stats"]),
            ("Статистика по поведению", result["behavior_stats"]),
        ]
        for title, data in sections:
            self.output.insert(tk.END, f"{title}:\n")
            if not data:
                self.output.insert(tk.END, "- нет данных\n")
            for name, count, pct in data:
                self.output.insert(tk.END, f"- {name}: {count} ({pct}%)\n")
            self.output.insert(tk.END, "\n")

        self.output.insert(tk.END, "UX/UI рекомендации:\n")
        for item in result["advice"]:
            self.output.insert(tk.END, f"- {item}\n")


if __name__ == "__main__":
    App().mainloop()
