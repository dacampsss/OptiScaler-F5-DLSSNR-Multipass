// x64 MSVC: cl /std:c++20 /EHsc /W4 /WX /O2 tests/nr_weights_per_model_smoke.cpp
// Neural Rendering model weights per model (OptiScaler/dlssnr/DlssNr_WeightsRules.h): migration of the old single
// WeightsSet, which list a folder appears in, the "Not used now" line, the rebuild key.
#include "../OptiScaler/dlssnr/DlssNr_WeightsRules.h"

#include <cstdio>

using namespace DlssNr::WeightsRules;

static int failures = 0;
#define CHECK(c)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(c))                                                                                                      \
        {                                                                                                              \
            printf("FAIL line %d: %s\n", __LINE__, #c);                                                                \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

int main()
{
    // File names tell the model; the dot keeps a longer resource name apart; case is ignored
    Models m;
    NoteFile(m, "WEIGHTS_HT.1A2B3C4D.bin");
    CHECK(m.a && !m.b);
    NoteFile(m, "WEIGHTS_HTX.1A2B3C4D.bin");
    CHECK(!m.b);
    NoteFile(m, "weights_cc_small_fp8.00000001.bin");
    CHECK(m.a && m.b);
    // Some modified DLSS-NR DLLs name Model A's resource WEIGHTS_PL
    Models alt;
    NoteFile(alt, "WEIGHTS_PL.0A0B0C0D.bin");
    CHECK(alt.a && !alt.b);
    NoteFile(alt, "WEIGHTS_PLX.0A0B0C0D.bin");
    CHECK(!alt.b);
    std::string altSet;
    CHECK(OldForwarderCanServe("H", "", alt, altSet) && altSet == "H"); // only A, the folder has only A
    Models none;
    NoteFile(none, "readme.txt");
    NoteFile(none, "WEIGHTS_HT");
    CHECK(!none.a && !none.b);

    // Migration: both models take the old set, whatever is on disk (the first config read runs before the
    // OptiScaler folder is known, so there is nothing to look at)
    Pair out;
    CHECK(Migrate("Strong", false, out) && out.a == "Strong" && out.b == "Strong");
    CHECK(!Migrate("Strong", true, out)); // a new key is in the file: nothing to migrate
    CHECK(!Migrate("", false, out));      // no old set
    CHECK(!Migrate("auto", false, out) && out.a.empty() && out.b.empty()); // "auto" = unset
    CHECK(!Migrate("AUTO", false, out));
    CHECK(Migrate("automatic", false, out) && out.a == "automatic"); // only the exact word is unset

    // The failure the first version had: it looked in the folder, found nothing at that time (folder not known yet)
    // and migrated to stock, which the next Save wrote over the player's set
    {
        const auto folderCheckedMigration = [](const std::string& oldSet, Models folder) -> Pair
        {
            Pair p;
            if (!oldSet.empty())
            {
                if (folder.a)
                    p.a = oldSet;
                if (folder.b)
                    p.b = oldSet;
            }
            return p;
        };
        const Pair oldResult = folderCheckedMigration("StrongB-SSS1", Models {}); // no folder access
        CHECK(oldResult.a.empty() && oldResult.b.empty());                         // the set was lost
        Migrate("StrongB-SSS1", false, out);
        CHECK(out.a == "StrongB-SSS1" && out.b == "StrongB-SSS1");                 // now kept for both
    }

    // An old forwarder (one folder for both models)
    std::string set;
    CHECK(OldForwarderCanServe("S", "S", {}, set) && set == "S");
    CHECK(OldForwarderCanServe("", "", {}, set) && set.empty());
    CHECK(!OldForwarderCanServe("S", "T", { true, true }, set));
    CHECK(OldForwarderCanServe("S", "", { true, false }, set) && set == "S");  // only A, folder has only A
    CHECK(!OldForwarderCanServe("S", "", { true, true }, set));                // folder would also serve B
    CHECK(OldForwarderCanServe("", "T", { false, true }, set) && set == "T");
    CHECK(!OldForwarderCanServe("", "T", { true, true }, set));
    CHECK(!OldForwarderCanServe("S", "", { false, true }, set));               // the folder has no file for A

    // Lists: a folder shows only in the list of a model it has a file for; both models -> both lists
    const std::vector<Set> sets = { { "BothSet", { true, true } }, { "OnlyA", { true, false } },
                                    { "OnlyB", { false, true } }, { "Empty", {} } };
    const auto listA = ListFor(sets, true);
    const auto listB = ListFor(sets, false);
    CHECK(listA.size() == 2 && listA[0] == "BothSet" && listA[1] == "OnlyA");
    CHECK(listB.size() == 2 && listB[0] == "BothSet" && listB[1] == "OnlyB");

    // "Not used now": only when a set is chosen and no pass runs that model
    CHECK(NotUsedNow(true, true, false, true));
    CHECK(!NotUsedNow(true, true, true, false));
    CHECK(NotUsedNow(true, false, true, false));
    CHECK(!NotUsedNow(true, false, false, true));
    CHECK(!NotUsedNow(false, true, false, false)); // stock: nothing to say

    // A change of either set changes the key
    CHECK(BuiltKey("a", "b") == "a|b");
    CHECK(BuiltKey("", "") == "|");
    CHECK(BuiltKey("a", "") != BuiltKey("", "a"));

    printf(failures ? "FAILED\n" : "PASS\n");
    return failures ? 1 : 0;
}
