import matplotlib.pyplot as plt

# =========================
# Données (Manhattan, pas de diagonales)
# =========================
coords = {
    1: (3, 8),
    2: (7, 0),
    3: (0, 2),
    4: (0, 5),
    5: (8, 4),
}

def manhattan(a, b):
    xa, ya = coords[a]
    xb, yb = coords[b]
    return abs(xa - xb) + abs(ya - yb)

# =========================
# Affichage d'une itération
# =========================
def draw_step(step_idx, tour_partial, current, unvisited, candidates, chosen,
              total_cost_so_far, pause=0.9, save_frames=False):
    """
    tour_partial : liste ordonnée des sommets déjà fixés (commence par [1,...])
    current     : sommet courant (dernier de tour_partial)
    unvisited   : ensemble des sommets non encore visités
    candidates  : liste [(v, dist), ...] triée par (dist, v)
    chosen      : sommet choisi v*
    total_cost_so_far : coût cumulé du chemin construit jusqu'ici (sans le retour final)
    """
    plt.figure(figsize=(7.2, 6.4))
    ax = plt.gca()
    ax.set_title(
        f"Étape {step_idx} — depuis {current}  |  coût cumulé = {total_cost_so_far}\n"
        "Bleu: gardé  ·  Gris: candidats  ·  Rouge: choisi"
    )
    ax.set_xlabel("x (cases)")
    ax.set_ylabel("y (cases)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    # nuage de points + labels
    xs_all = [p[0] for p in coords.values()]
    ys_all = [p[1] for p in coords.values()]
    ax.scatter(xs_all, ys_all, s=40, zorder=5)
    for k, (x, y) in coords.items():
        ax.text(x, y, str(k), fontsize=10, ha="right", va="bottom")

    # 1) tracer le chemin déjà construit (bleu)
    if len(tour_partial) >= 2:
        for i in range(len(tour_partial) - 1):
            a, b = tour_partial[i], tour_partial[i + 1]
            xa, ya = coords[a]
            xb, yb = coords[b]
            ax.plot([xa, xb], [ya, yb], color="#2b78e4", linewidth=2.2, zorder=3)

    # 2) tracer toutes les candidates depuis current (gris) + distances annotées
    for v, dist in candidates:
        xc, yc = coords[current]
        xv, yv = coords[v]
        ax.plot([xc, xv], [yc, yv], color="#999999", linewidth=1.2, alpha=0.55, zorder=1)
        mx, my = (xc + xv) / 2, (yc + yv) / 2
        ax.text(mx, my, f"{dist}", fontsize=9, color="#555555")

    # 3) tracer l'arête choisie en rouge (par-dessus)
    if chosen is not None:
        xc, yc = coords[current]
        xv, yv = coords[chosen]
        ax.plot([xc, xv], [yc, yv], color="#e32020", linewidth=2.8, alpha=0.95, zorder=4)

    # cadre confortable
    pad = 1.0
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
    ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)

    plt.tight_layout()
    if save_frames:
        plt.savefig(f"etape_{step_idx:02d}.png", dpi=140)
    plt.show(block=False)
    plt.pause(pause)
    plt.close()

# =========================
# Construction gloutonne depuis 1 (visualisée étape par étape)
# =========================
def greedy_cycle_from_1_visual(pause=0.9, save_frames=False, verbose=True):
    start = 1
    unvisited = set(coords.keys())
    unvisited.remove(start)
    tour = [start]
    total = 0
    current = start

    step = 1
    while unvisited:
        # liste des candidats (sommet, distance) et tri
        candidates = [(v, manhattan(current, v)) for v in unvisited]
        candidates.sort(key=lambda t: (t[1], t[0]))  # on casse l'égalité par l'id

        chosen, dist = candidates[0]

        if verbose:
            print(f"Étape {step} — depuis {current} @ {coords[current]}")
            for v, dd in candidates:
                print(f"  candidate: {current}->{v}  d={dd}")
            print(f"  ==> choisi: {current}->{chosen}  (d={dist})\n")

        # affichage de l'étape avant d'entériner le choix
        draw_step(step_idx=step,
                  tour_partial=tour,
                  current=current,
                  unvisited=unvisited,
                  candidates=candidates,
                  chosen=chosen,
                  total_cost_so_far=total,
                  pause=pause,
                  save_frames=save_frames)

        # on entérine le choix
        total += dist
        tour.append(chosen)
        unvisited.remove(chosen)
        current = chosen
        step += 1

    # dernière étape : retour au départ (on montre les candidats = juste le retour)
    ret = manhattan(current, start)
    if verbose:
        print(f"Retour au départ: {current}->{start}  (d={ret})")

    # affichage du retour (candidates = uniquement retour)
    draw_step(step_idx=step,
              tour_partial=tour,         # déjà toutes les arêtes gardées
              current=current,
              unvisited=set(),           # plus de non-visités
              candidates=[(start, ret)], # montrer le retour en gris + rouge
              chosen=start,
              total_cost_so_far=total,
              pause=pause,
              save_frames=save_frames)

    total += ret

    if verbose:
        print(f"Chemin final: {tour + [start]}")
        print(f"Distance totale (Manhattan): {total}")

    # trace final figée du cycle complet
    plt.figure(figsize=(7.2, 6.4))
    ax = plt.gca()
    cyc = tour + [start]
    xs = [coords[i][0] for i in cyc]
    ys = [coords[i][1] for i in cyc]
    ax.plot(xs, ys, color="#2b78e4", linewidth=2.4, marker="o")
    for k, (x, y) in coords.items():
        ax.text(x, y, str(k), fontsize=10, ha="right", va="bottom")
    # annoter chaque arête
    for i in range(len(cyc) - 1):
        a, b = cyc[i], cyc[i+1]
        xa, ya = coords[a]
        xb, yb = coords[b]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my, f"{manhattan(a,b)}", fontsize=9, color="#444444")
    ax.set_title(f"Cycle final glouton — distance totale = {total}")
    ax.set_xlabel("x (cases)")
    ax.set_ylabel("y (cases)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    pad = 1.0
    xs_all = [p[0] for p in coords.values()]
    ys_all = [p[1] for p in coords.values()]
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
    ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)
    plt.tight_layout()
    plt.show()

    return tour, total

# =========================
# Exécution
# =========================
if __name__ == "__main__":
    # Régle la vitesse (en secondes) et sauvegarde optionnelle des frames
    PAUSE = 5.0
    SAVE_FRAMES = False   # passe à True pour enregistrer etape_01.png, etape_02.png, ...

    tour, total = greedy_cycle_from_1_visual(pause=PAUSE, save_frames=SAVE_FRAMES, verbose=True)