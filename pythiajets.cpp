#include "fastjet_compat.h"
#include "Pythia8/Pythia.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>

using namespace Pythia8;
using namespace fastjet;

// Function to display progress bar
void printProgress(float progress) {
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    clock_t cpu_start = clock();

    // Parse number of events from command line
    int nEvent = 100; // default value
    if (argc > 1) {
        nEvent = std::atoi(argv[1]);
    }
    std::cout << "Running with " << nEvent << " events" << std::endl;

    // Create ROOT histograms
    TH1F* hJetPt = new TH1F("hJetPt", "Jet p_{T};p_{T} (GeV/c);Counts", 100, 0, 100);
    TH1F* hJetEta = new TH1F("hJetEta", "Jet #eta;#eta;Counts", 100, -5, 5);

    // Initialize Pythia
    Pythia pythia;
    
    // Configure Pythia for pp collisions at 13 TeV
    pythia.readString("Beams:idA = 2212");     // proton
    pythia.readString("Beams:idB = 2212");     // proton
    pythia.readString("Beams:eCM = 13000.");   // 13 TeV
    pythia.readString("SoftQCD:inelastic = on");  // Turn on inelastic processes
    pythia.readString("HardQCD:all = on");       // Turn on all hard QCD processes
    pythia.readString("Print:quiet = on");        // Reduce Pythia output
    pythia.init();

    // FastJet setup
    double R = 0.4;
    JetDefinition jet_def(antikt_algorithm, R);
    AreaDefinition area_def(active_area_explicit_ghosts);

    // Event loop
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
        if (!pythia.next()) continue;

        // Vector to store final-state particles for jet reconstruction
        std::vector<PseudoJet> particles;

        // Loop over all particles in the event
        for (int i = 0; i < pythia.event.size(); ++i) {
            if (pythia.event[i].isFinal() && pythia.event[i].isHadron()) {
                double eta = pythia.event[i].eta();
                double pt = pythia.event[i].pT();
                if (std::abs(eta) < 0.5 && pt > 0.2) {
                    particles.push_back(PseudoJet(
                        pythia.event[i].px(),
                        pythia.event[i].py(),
                        pythia.event[i].pz(),
                        pythia.event[i].e()
                    ));
                }
            }
        }

        // Perform jet clustering with area
        ClusterSequenceArea cs(particles, jet_def, area_def);
        vector<PseudoJet> jets = sorted_by_pt(cs.inclusive_jets(5.0));

        // Fill histograms
        for (const auto& jet : jets) {
            hJetPt->Fill(jet.pt());
            hJetEta->Fill(jet.eta());
        }

        // Update progress bar
        printProgress(float(iEvent + 1) / nEvent);
    }
    std::cout << std::endl; // New line after progress bar

    // Create output file and save histograms
    TFile* outFile = new TFile("jet_spectra.root", "RECREATE");
    
    // Save histograms
    hJetPt->Write();
    hJetEta->Write();
    outFile->Close();

    // Print final statistics
    pythia.stat();

    // Calculate and print timing information
    auto end_time = std::chrono::high_resolution_clock::now();
    clock_t cpu_end = clock();

    double cpu_time = static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    auto wall_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "\nTiming Information:" << std::endl;
    std::cout << "CPU time: " << cpu_time << " seconds" << std::endl;
    std::cout << "Wall time: " << wall_time << " seconds" << std::endl;
    std::cout << "CPU/Wall: " << cpu_time/wall_time << std::endl;

    return 0;
}