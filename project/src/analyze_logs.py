import pandas as pd
import json

df = pd.read_csv('logs/run_qwen35_gguf_math_tools_v2_all_competitions.csv')

print("="*90)
print("ANALISI DOMANDE - run_qwen35_gguf_math_tools_v2")
print("="*90)

# Metriche generali
print(f"\nTotale righe: {len(df)}")
print(f"Competizioni: {sorted(df['competition_name'].unique())}")

# Per competizione
print("\n" + "="*90)
print("BREAKDOWN PER COMPETIZIONE")
print("="*90)

for comp in sorted(df['competition_name'].unique()):
    comp_df = df[df['competition_name'] == comp]
    correct = comp_df['correct'].sum()
    total = len(comp_df)
    acc = comp_df['correct'].mean() if total > 0 else 0
    timeout = comp_df['timed_out'].sum()
    lat_mean = comp_df['latency_seconds'].mean()
    lat_max = comp_df['latency_seconds'].max()

    print(f"\n{comp}:")
    print(f"  Domande: {total:3d} | Corrette: {correct:3d} | Accuracy: {acc:5.1%}")
    print(f"  Latenza: media {lat_mean:.2f}s, max {lat_max:.2f}s")
    print(f"  Timeout: {timeout}")

# Strategie usate
print("\n" + "="*90)
print("STRATEGIE USATE")
print("="*90)

strat_summary = df.groupby('strategy').agg({
    'correct': ['sum', 'count'],
    'latency_seconds': 'mean',
    'timed_out': 'sum'
}).round(2)

strat_summary.columns = ['Corrette', 'Totale', 'Latenza', 'Timeout']
strat_summary['Acc%'] = (strat_summary['Corrette'] / strat_summary['Totale'] * 100).round(1)
strat_summary = strat_summary[['Totale', 'Corrette', 'Acc%', 'Latenza', 'Timeout']]

print(strat_summary.to_string())

# Campioni di domande
print("\n" + "="*90)
print("ESEMPI DI DOMANDE PER COMPETIZIONE")
print("="*90)

for comp in sorted(df['competition_name'].unique()):
    comp_df = df[df['competition_name'] == comp].drop_duplicates('question_text')
    print(f"\n{comp.upper()}:")

    for idx, (_, row) in enumerate(comp_df.head(3).iterrows(), 1):
        q_text = row['question_text'][:90]
        strategy = row['strategy']
        correct = "✓" if row['correct'] else "✗"
        lat = f"{row['latency_seconds']:.2f}s"

        print(f"  {idx}. [{correct}] {q_text}...")
        print(f"     Strategy: {strategy} | Latency: {lat}")

# Errori e fallimenti
print("\n" + "="*90)
print("ANALISI ERRORI")
print("="*90)

errors = df[df['correct'] == False]
print(f"\nDomande corrette: {df['correct'].sum()}/{len(df)} ({df['correct'].mean():.1%})")
print(f"Domande sbagliate: {len(errors)}")

if len(errors) > 0:
    print(f"\nMotivi fallimento:")
    print(f"  - Timeout: {errors['timed_out'].sum()}")
    print(f"  - Parse error: {(errors['error_message'].notna()).sum()}")
    wrong = len(errors) - errors['timed_out'].sum() - (errors['error_message'].notna()).sum()
    print(f"  - Risposta sbagliata: {wrong}")

    # Errori per competizione
    print(f"\nErrori per competizione:")
    for comp in sorted(df['competition_name'].unique()):
        comp_errors = errors[errors['competition_name'] == comp]
        if len(comp_errors) > 0:
            print(f"  - {comp}: {len(comp_errors)} errori")

            # Esempi di fallimenti
            for _, row in comp_errors.head(2).iterrows():
                q = row['question_text'][:70]
                print(f"      Q: {q}...")
