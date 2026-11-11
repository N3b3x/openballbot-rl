# compact finished training .out files: drop the three Warp hfield-overflow warning lines, keep everything else
cd /c/Users/Owner/Documents/GitHub/openballbot-rl/outputs/study/warp_ppo
for f in $(ls -S *.out); do
  n=${f%.out}
  case "$n" in g25_ret_rs025_stand_s0|g25_ret_rs025_stand_s1) continue;; esac
  [ $(stat -c %s "$f") -lt 50000000 ] && continue
  grep -v -e "warn_overflow" -e "height field collision overflow" -e "decrease the number of hfield rows" "$f" > "$f.tmp" && mv -f "$f.tmp" "$f" && echo "compacted $f $(stat -c %s "$f")"
done
echo COMPACT_DONE
